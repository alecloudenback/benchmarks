using Reactant

@isdefined(NSTEPS) || include(joinpath(@__DIR__, "common.jl"))

# BasicTerm_ME traced to StableHLO and compiled by XLA via Reactant.jl.
#
# This is the counterpart to `term_me_iterative_jax.py`: the same model, the same
# scan-shaped loop, compiled by the same backend. The only thing that differs is
# the frontend language, which is the point - it separates "how fast is XLA on
# this model" from "how fast is the Python/JAX frontend".
#
# The model is written in ordinary Julia array style. Reactant traces it once,
# `@trace while` lowers the 277-month loop into a StableHLO `while` (the analogue
# of `lax.scan`, rather than unrolling 277 copies of the body), and XLA fuses the
# whole thing into one program.
#
# Two things about tracing that are easy to get wrong:
#
#   * Scalars carried across a `@trace while` must already be traced values.
#     A plain `total = 0.0` is treated as a constant, the loop still runs, and
#     the accumulated value is silently discarded - you get 0.0 back, not an
#     error. Hence `promote_to_traced`.
#   * `fld` on traced integers currently lowers to truncating division. That is
#     immaterial here: it only differs for `duration_mth + t < 0`, i.e. new
#     business that has not yet been written, where policies in force are zero
#     and every affected term is multiplied by that zero. The reference check
#     below is what actually confirms it.

const promote_to_traced = Reactant.ReactantCore.promote_to_traced

# The rate-conversion mode has to be resolved by dispatch on a value passed into
# the traced function, not by a branch written inside the loop: `@trace` lifts
# every local it sees in the body, and a plain `Bool` gets lifted into a
# `TracedRNumber{Bool}`, which then cannot be used as a condition. A `Val` is not
# traceable, so it survives as a compile-time constant.
to_monthly_mort(::Val{:inline}, q) = 1 .- (1 .- q) .^ (1 / 12)
to_monthly_mort(::Val{:table}, q) = q

to_monthly_lapse(::Val{:inline}, dur, tbl) =
    1 .- (1 .- max.(0.02, 0.1 .- 0.02 .* dur)) .^ (1 / 12)
to_monthly_lapse(::Val{:table}, dur, tbl) = tbl[clamp.(dur, 0, 4) .+ 1]

function init_backend!(force_cpu::Bool)
    if !force_cpu
        try
            Reactant.set_default_backend("gpu")
        catch e
            @warn "Reactant GPU backend unavailable, falling back to CPU" exception = e
        end
    end
    println("device=", Reactant.XLA.platform_name(Reactant.XLA.default_backend()))
    return nothing
end

ENTRYPOINTS[:init] = init_backend!

"""
    project_reactant(...)

A transcription of the JAX `iterative_core` scan body. `mort_flat` is the
mortality table flattened column-major so the per-model-point lookup is a single
gather; with `rates=:inline` it holds annual rates that are converted in the
loop (what JAX does), with `rates=:table` it holds pre-converted monthly rates.
"""
function project_reactant(premium_pp, duration_mth, age_at_entry, sum_assured,
                          policy_count, policy_term, mort_flat, lapse_mth,
                          disc_v, infl_v, mode::Val)
    pols_lapse = zero(policy_count)
    pols_death = zero(policy_count)
    pols_if = ifelse.(duration_mth .> 0, policy_count, 0.0)
    total = promote_to_traced(0.0)
    t = promote_to_traced(0)

    @trace while t < NSTEPS
        dm = duration_mth .+ t
        dur = fld.(dm, 12)
        age = age_at_entry .+ dur

        init = pols_if .- pols_lapse .- pols_death
        maturity = ifelse.(dm .== policy_term .* 12, init, 0.0)
        new_biz = ifelse.(dm .== 0, policy_count, 0.0)
        bef_decr = init .- maturity .+ new_biz

        # Column-major flat index into the 103x6 table.
        lin = clamp.(dur, 0, 5) .* 103 .+ clamp.(age .- 18, 0, 102) .+ 1
        q = mort_flat[lin]
        mort_mth = to_monthly_mort(mode, q)
        death = bef_decr .* mort_mth

        disc = sum(Reactant.Ops.dynamic_slice(disc_v, [t + 1], [1]))
        infl = sum(Reactant.Ops.dynamic_slice(infl_v, [t + 1], [1]))

        premiums = premium_pp .* bef_decr
        claims = sum_assured .* death
        commissions = ifelse.(dur .== 0, premiums, 0.0)
        expenses = EXPENSE_ACQ .* new_biz .+ bef_decr .* (EXPENSE_MAINT / 12) .* infl

        total += sum(premiums .- claims .- expenses .- commissions) * disc

        pols_lapse = (bef_decr .- death) .* to_monthly_lapse(mode, dur, lapse_mth)
        pols_death = death
        pols_if = bef_decr
        t += 1
    end
    return total
end

"""
    reactant_inputs(tables, multiplier, rates)

Build the device arrays. Tiling happens on the host and the result is moved
across in one go, matching what `np.tile` + `jnp.array` do in the JAX version.
Integers are `Int64` for the same reason - the JAX model runs under
`jax_enable_x64`.
"""
function reactant_inputs(tables, multiplier::Integer, rates::Symbol)
    mp = copy(tables.mp)
    rate = Dict((r.age_at_entry, r.policy_term) => r.premium_rate
                for r in eachrow(tables.prem))
    mp.premium_rate = [rate[(r.age_at_entry, r.policy_term)] for r in eachrow(mp)]
    sort!(mp, :policy_id)

    R = Reactant.to_rarray
    f64(v) = R(tile(Float64.(v), multiplier))
    i64(v) = R(tile(Int64.(v), multiplier))

    mort_ann = Matrix{Float64}(tables.mort[:, 2:end])           # 103 x 6
    mort = rates === :inline ? mort_ann : @.(1 - (1 - mort_ann)^(1 / 12))
    lapse_mth = [1 - (1 - max(0.02, 0.1 - 0.02 * d))^(1 / 12) for d in 0:4]

    zero_spot = Float64.(tables.disc.zero_spot)

    return (
        f64(round.(Float64.(mp.sum_assured) .* mp.premium_rate; digits = 2)),
        i64(mp.duration_mth),
        i64(mp.age_at_entry),
        f64(mp.sum_assured),
        f64(mp.policy_count),
        i64(mp.policy_term),
        R(vec(mort)),
        R(lapse_mth),
        R([(1 + zero_spot[t ÷ 12 + 1])^(-t / 12) for t in 0:(NSTEPS - 1)]),
        R([(1 + INFLATION_RATE)^(t / 12) for t in 0:(NSTEPS - 1)]),
        Val(rates),
    )
end

function time_reactant(multiplier::Integer, runs::Integer, rates::Symbol = :inline)
    tables = read_tables()
    args = reactant_inputs(tables, multiplier, rates)

    # Compilation is shape-specialised, so it has to happen at the real problem
    # size. Compile, run once to warm up, then time - exactly the shape of the
    # JAX runner's `block_until_ready()` warm-up.
    compiled = @compile project_reactant(args...)
    run() = Float64(compiled(args...))
    run()

    result, seconds = time_run(run)
    for _ in 2:runs
        r, s = time_run(run)
        result, seconds = r, min(seconds, s)
    end

    n = length(args[2])
    report("Julia Reactant/XLA model (rates=$rates)", n, result, seconds, multiplier)
    return result, seconds
end

ENTRYPOINTS[:run] = time_reactant
