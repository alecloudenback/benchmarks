@isdefined(on_gpu) || include(joinpath(@__DIR__, "cuda_backend.jl"))

# BasicTerm_ME as a single hand-written CUDA kernel.
#
# One thread owns one model point and walks it through its own lifetime,
# accumulating the discounted net cash flow in a register. Compared with the
# array formulation this changes the cost structure completely:
#
#   * the 100M-element policy state (in force / deaths / lapses) never touches
#     global memory - it lives in registers for the duration of the projection,
#     so the model point data is read exactly once instead of ~277 times;
#   * each thread only iterates over the months in which its own policy is
#     actually in force (`max(0, -duration_mth)` up to maturity) rather than all
#     277 months of the longest policy in the portfolio;
#   * `mort_rate` and `lapse_rate` only change at policy anniversaries, so the
#     month loop is nested inside a policy-year loop and the rate lookups happen
#     ~1/12 as often;
#   * the discount and inflation factors are the same for every thread, so they
#     are staged into shared memory once per block.
#
# `policy_pv` is backend-agnostic so the identical arithmetic can be run on the
# CPU for validation.

@inline function policy_pv(i, premium_pp, duration_mth, age_at_entry, sum_assured,
                           policy_count, policy_term, mort, lapse, disc, infl_disc, mode)
    @inbounds begin
        dm0 = duration_mth[i]
        term = policy_term[i]
        entry_age = age_at_entry[i]
        count = policy_count[i]
        face = sum_assured[i]
        prem_pp = premium_pp[i]

        # First month in which the policy exists (`duration_mth` is negative for
        # new business written during the projection) and the month it matures.
        t = max(zero(dm0), -dm0)
        t_end = Int32(12) * term - dm0

        lives = count
        total = 0.0
        while t < t_end
            dm = dm0 + t
            dur = dm ÷ Int32(12)
            # Assumption lookups, hoisted out of the month loop.
            qm = mort_rate_mth(mode, mort, entry_age + dur - Int32(17),
                               min(dur, Int32(5)) + Int32(1))
            lm = lapse_rate_mth(mode, lapse, dur)
            first_year = dur == zero(dur)
            # Stop at the earlier of maturity and the next policy anniversary.
            t_stop = min(t_end, t + Int32(12) - dm % Int32(12))
            while t < t_stop
                deaths = lives * qm
                premiums = prem_pp * lives
                cf = premiums - face * deaths
                first_year && (cf -= premiums)                    # commissions
                (dm0 + t) == zero(dm0) && (cf -= EXPENSE_ACQ * count)
                total += cf * disc[t + Int32(1)] -
                         lives * (EXPENSE_MAINT / 12) * infl_disc[t + Int32(1)]
                lives -= deaths + (lives - deaths) * lm
                t += Int32(1)
            end
        end
        return total
    end
end

function project_kernel!(partials, premium_pp, duration_mth, age_at_entry, sum_assured,
                         policy_count, policy_term, mort, lapse, disc, infl_disc,
                         n::Int, mode)
    s_disc = CuStaticSharedArray(Float64, NSTEPS)
    s_infl = CuStaticSharedArray(Float64, NSTEPS)
    j = threadIdx().x
    while j <= NSTEPS
        @inbounds s_disc[j] = disc[j]
        @inbounds s_infl[j] = infl_disc[j]
        j += blockDim().x
    end
    sync_threads()

    gid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    acc = 0.0
    i = gid
    while i <= n
        acc += policy_pv(i, premium_pp, duration_mth, age_at_entry, sum_assured,
                         policy_count, policy_term, mort, lapse, s_disc, s_infl, mode)
        i += stride
    end
    @inbounds partials[gid] = acc
    return nothing
end

struct KernelWorkspace{V <: AbstractVector{Float64}}
    disc::V
    infl_disc::V
    partials::V
    threads::Int
    blocks::Int
end

"""
    workspace(mp, assume)

Pick a launch configuration and allocate the per-thread accumulator. The grid is
sized for full occupancy rather than one block per model point: threads take
policies with a grid stride, which averages out the wide spread in per-policy
loop lengths across the whole grid.
"""
function workspace(mp::ModelPoints, assume::Assumptions, mode::Val = Val(:table))
    n = length(mp)
    disc = device(assume.disc)
    infl_disc = device(assume.infl .* assume.disc)
    if !on_gpu()
        nthreads = Threads.nthreads()
        return KernelWorkspace(disc, infl_disc, zeros(nthreads), 1, nthreads)
    end
    args = (CUDA.zeros(Float64, 1), mp.premium_pp, mp.duration_mth, mp.age_at_entry,
            mp.sum_assured, mp.policy_count, mp.policy_term, mort_table(assume, mode),
            assume.lapse_mth, disc, infl_disc, n, mode)
    kernel = @cuda launch = false project_kernel!(args...)
    config = launch_configuration(kernel.fun)
    threads = config.threads
    blocks = min(cld(n, threads), config.blocks)
    return KernelWorkspace(disc, infl_disc, CUDA.zeros(Float64, threads * blocks),
                           threads, blocks)
end

function project(mp::ModelPoints, assume::Assumptions, ws::KernelWorkspace,
                 mode::Val = Val(:table))
    args = (ws.partials, mp.premium_pp, mp.duration_mth, mp.age_at_entry, mp.sum_assured,
            mp.policy_count, mp.policy_term, mort_table(assume, mode), assume.lapse_mth,
            ws.disc, ws.infl_disc, length(mp), mode)
    if on_gpu()
        @cuda threads = ws.threads blocks = ws.blocks project_kernel!(args...)
    else
        project_threaded!(args...)
    end
    return sum(ws.partials)
end

# CPU mirror of the kernel, used for validation without a GPU.
function project_threaded!(partials, premium_pp, duration_mth, age_at_entry, sum_assured,
                           policy_count, policy_term, mort, lapse, disc, infl_disc,
                           n::Int, mode)
    nt = length(partials)
    Threads.@threads for k in 1:nt
        acc = 0.0
        i = k
        while i <= n
            acc += policy_pv(i, premium_pp, duration_mth, age_at_entry, sum_assured,
                             policy_count, policy_term, mort, lapse, disc, infl_disc, mode)
            i += nt
        end
        partials[k] = acc
    end
    return nothing
end

function time_kernel(multiplier::Integer, runs::Integer, rates::Symbol = :table)
    mode = Val(rates)
    tables = read_tables()
    assume = assumptions(tables)

    warm_mp = model_points(tables, 1)
    project(warm_mp, assume, workspace(warm_mp, assume, mode), mode)

    mp = model_points(tables, multiplier)
    ws = workspace(mp, assume, mode)
    result, seconds = time_run(() -> project(mp, assume, ws, mode))
    for _ in 2:runs
        r, s = time_run(() -> project(mp, assume, ws, mode))
        result, seconds = r, min(seconds, s)
    end

    report("Julia CUDA kernel model (rates=$rates)", length(mp), result, seconds, multiplier)
    return result, seconds
end

ENTRYPOINTS[:run] = time_kernel
