@isdefined(on_gpu) || include(joinpath(@__DIR__, "cuda_backend.jl"))

# BasicTerm_ME as a time-stepped array model.
#
# This is the direct analogue of the JAX `lax.scan` implementation: the state is
# three vectors of length `n` (one entry per model point) and the projection is a
# loop over 277 months, each month issuing a handful of fused elementwise kernels
# plus one reduction. Every model point is advanced through every month whether or
# not it is in force, because the whole portfolio moves in lockstep.
#
# With the default `rates=:inline` the arithmetic matches the JAX scan
# statement for statement, including recomputing `1 - (1 - q)^(1/12)` over the
# whole portfolio in every month. `rates=:table` swaps in pre-converted monthly
# rates and is there to measure what that costs.
#
# The per-element bodies are plain Julia functions so that `.` fusion collapses
# each expression into a single kernel, and so the same code runs on `Array`.
# Lookup tables ride along as `Ref`s, which broadcasting treats as scalars and
# CUDA.jl adapts into a device-side reference holding a `CuDeviceArray`.

@inline function bef_decr(if_prev, lapse_prev, death_prev, dm0, term, count, t)
    dm = dm0 + t
    init = if_prev - lapse_prev - death_prev
    maturity = ifelse(dm == Int32(12) * term, init, zero(init))
    new_biz = ifelse(dm == zero(dm), count, zero(count))
    return init - maturity + new_biz
end

@inline function pols_death(bd, dm0, age_at_entry, t, mort, mode)
    dur = fld(dm0 + t, Int32(12))
    q = mort_rate_mth(mode, mort,
                      clamp(age_at_entry + dur - Int32(18), Int32(0), Int32(102)) + 1,
                      clamp(dur, Int32(0), Int32(5)) + 1)
    return bd * q
end

@inline function pols_lapse(bd, death, dm0, t, lapse, mode)
    dur = fld(dm0 + t, Int32(12))
    return (bd - death) * lapse_rate_mth(mode, lapse, dur)
end

@inline function net_cf(bd, death, dm0, count, premium_pp, sum_assured, t, infl)
    dm = dm0 + t
    dur = fld(dm, Int32(12))
    premiums = premium_pp * bd
    claims = sum_assured * death
    commissions = ifelse(dur == zero(dur), premiums, zero(premiums))
    new_biz = ifelse(dm == zero(dm), count, zero(count))
    expenses = EXPENSE_ACQ * new_biz + bd * (EXPENSE_MAINT / 12) * infl
    return premiums - claims - expenses - commissions
end

struct ArrayState{V <: AbstractVector{Float64}}
    pols_if::V      # policies in force at BEF_DECR, previous month
    pols_lapse::V
    pols_death::V
end

function ArrayState(mp::ModelPoints)
    ArrayState(
        ifelse.(mp.duration_mth .> 0, mp.policy_count, 0.0),
        zero(mp.policy_count),
        zero(mp.policy_count),
    )
end

function reset!(st::ArrayState, mp::ModelPoints)
    st.pols_if .= ifelse.(mp.duration_mth .> 0, mp.policy_count, 0.0)
    fill!(st.pols_lapse, 0.0)
    fill!(st.pols_death, 0.0)
    return st
end

function project_array(mp::ModelPoints, assume::Assumptions, st::ArrayState,
                       mode::Val = Val(:inline))
    mort = Ref(mort_table(assume, mode))
    lapse = Ref(assume.lapse_mth)
    total = 0.0
    for t in 0:(NSTEPS - 1)
        ti = Int32(t)
        infl = @inbounds assume.infl[t + 1]
        disc = @inbounds assume.disc[t + 1]

        # Roll forward to policies in force before decrements. Writing into
        # `pols_if` is safe: the update is elementwise and both decrement
        # vectors are dead after this statement.
        st.pols_if .= bef_decr.(st.pols_if, st.pols_lapse, st.pols_death,
                                mp.duration_mth, mp.policy_term, mp.policy_count, ti)
        st.pols_death .= pols_death.(st.pols_if, mp.duration_mth, mp.age_at_entry, ti,
                                     mort, mode)

        # Fused into the reduction, so no length-n temporary is materialised.
        cf = Broadcast.instantiate(Broadcast.broadcasted(net_cf,
            st.pols_if, st.pols_death, mp.duration_mth, mp.policy_count,
            mp.premium_pp, mp.sum_assured, ti, infl))
        total += mapreduce(identity, +, cf) * disc

        st.pols_lapse .= pols_lapse.(st.pols_if, st.pols_death, mp.duration_mth, ti,
                                     lapse, mode)
    end
    return total
end

function time_array(multiplier::Integer, runs::Integer, rates::Symbol = :inline)
    mode = Val(rates)
    tables = read_tables()
    assume = assumptions(tables)

    # Warm up on a single copy of the table so Julia's JIT and the CUDA module
    # load happen outside the measured region, exactly like the PyTorch runner.
    warm_mp = model_points(tables, 1)
    project_array(warm_mp, assume, ArrayState(warm_mp), mode)

    mp = model_points(tables, multiplier)
    st = ArrayState(mp)
    result, seconds = time_run(() -> project_array(mp, assume, st, mode))
    for _ in 2:runs
        reset!(st, mp)
        r, s = time_run(() -> project_array(mp, assume, st, mode))
        result, seconds = r, min(seconds, s)
    end

    report("Julia array model (rates=$rates)", length(mp), result, seconds, multiplier)
    return result, seconds
end

ENTRYPOINTS[:run] = time_array
