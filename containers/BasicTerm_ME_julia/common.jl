using DataFrames
using XLSX

# --- model constants -------------------------------------------------------
# The projection runs for `NSTEPS` months, which is
# `maximum(12 * policy_term - duration_mth) + 1` for lifelib's BasicTerm_ME
# model point file. Kept as a constant so it matches the JAX implementation,
# which hard-codes a scan length of 277.
const NSTEPS = 277
const EXPENSE_ACQ = 300.0
const EXPENSE_MAINT = 60.0
const INFLATION_RATE = 0.01

# --- backend ---------------------------------------------------------------
# This file is backend-agnostic on purpose: `main.jl` loads exactly one backend
# per process, so a Reactant run never loads CUDA.jl and vice versa. A backend
# installs itself by setting these hooks in its `init_backend!`.
#
# ADAPT     : host array -> device array (`identity` keeps everything on the CPU)
# TIME_RUN  : backend-specific timer, or `nothing` for the wall clock
const ADAPT = Ref{Any}(identity)
const TIME_RUN = Ref{Any}(nothing)

# A backend registers its `init_backend!` and timing entry point here. `main.jl`
# reads the function objects out of this dict rather than looking bindings up by
# name, which would trip Julia 1.12's world-age rules for bindings created by an
# `include` that happens inside a running function.
const ENTRYPOINTS = Dict{Symbol, Any}()

device(x::AbstractArray) = ADAPT[](x)

"""
    tile(x, m)

Repeat `x` end-to-end `m` times (`np.tile` semantics). Done on-device with
`copyto!` so the 100M-element vectors are never materialised in host memory.
"""
function tile(x::AbstractVector, m::Integer)
    m == 1 && return x
    n = length(x)
    out = similar(x, n * m)
    for k in 0:(m - 1)
        copyto!(out, k * n + 1, x, 1, n)
    end
    return out
end

# --- data ------------------------------------------------------------------
struct ModelPoints{VF <: AbstractVector{Float64}, VI <: AbstractVector{<:Integer}}
    premium_pp::VF
    duration_mth::VI
    age_at_entry::VI
    sum_assured::VF
    policy_count::VF
    policy_term::VI
end

Base.length(mp::ModelPoints) = length(mp.duration_mth)

struct Assumptions{M <: AbstractMatrix{Float64}, V <: AbstractVector{Float64}}
    mort_ann::M       # 103x6 annual mortality rate, as loaded, rows are ages 18:120
    mort_mth::M       # 103x6 monthly mortality rate, 1 - (1 - q)^(1/12)
    lapse_mth::V      # 5-element monthly lapse rate, indexed by min(duration, 4)
    disc::Vector{Float64}       # host: monthly discount factor, t = 0:NSTEPS-1
    infl::Vector{Float64}       # host: expense inflation factor, t = 0:NSTEPS-1
end

const DATA_DIR = joinpath(@__DIR__, "BasicTerm_ME")

function read_tables(dir = DATA_DIR)
    mp = DataFrame(XLSX.readtable(joinpath(dir, "model_point_table.xlsx"), 1))
    prem = DataFrame(XLSX.readtable(joinpath(dir, "premium_table.xlsx"), 1))
    mort = DataFrame(XLSX.readtable(joinpath(dir, "mort_table.xlsx"), 1))
    disc = DataFrame(XLSX.readtable(joinpath(dir, "disc_rate_ann.xlsx"), 1))

    # premium_table.xlsx has a two-level index; the outer level is only written
    # on the first row of each group, so forward-fill it.
    let last = 0
        prem.age_at_entry = map(v -> ismissing(v) ? last : (last = v), prem.age_at_entry)
    end
    return (; mp, prem, mort, disc)
end

function model_points(tables, multiplier::Integer)
    mp = copy(tables.mp)
    rate = Dict((r.age_at_entry, r.policy_term) => r.premium_rate for r in eachrow(tables.prem))
    mp.premium_rate = [rate[(r.age_at_entry, r.policy_term)] for r in eachrow(mp)]
    sort!(mp, :policy_id)

    f64(v) = device(Float64.(v))
    i32(v) = device(Int32.(v))

    return ModelPoints(
        tile(f64(round.(Float64.(mp.sum_assured) .* mp.premium_rate; digits = 2)), multiplier),
        tile(i32(mp.duration_mth), multiplier),
        tile(i32(mp.age_at_entry), multiplier),
        tile(f64(mp.sum_assured), multiplier),
        tile(f64(mp.policy_count), multiplier),
        tile(i32(mp.policy_term), multiplier),
    )
end

"""
    assumptions(tables)

Build the assumption set. Both the annual rates as loaded and their monthly
equivalents are kept: annual-to-monthly conversion is a function of small lookup
tables only (103x6 mortality cells and 5 distinct lapse rates), so a model can
either convert once here or repeat the `^(1/12)` over the whole portfolio in
every projection month. See `rate_mode` in the two models.
"""
function assumptions(tables)
    mort_ann = Matrix{Float64}(tables.mort[:, 2:end])           # 103 x 6
    mort_mth = @. 1 - (1 - mort_ann)^(1 / 12)
    lapse_ann = [max(0.02, 0.1 - 0.02 * d) for d in 0:4]
    lapse_mth = @. 1 - (1 - lapse_ann)^(1 / 12)

    zero_spot = Float64.(tables.disc.zero_spot)
    disc = [(1 + zero_spot[t ÷ 12 + 1])^(-t / 12) for t in 0:(NSTEPS - 1)]
    infl = [(1 + INFLATION_RATE)^(t / 12) for t in 0:(NSTEPS - 1)]

    return Assumptions(device(mort_ann), device(mort_mth), device(lapse_mth), disc, infl)
end

# --- rate conversion mode --------------------------------------------------
# `:inline` reproduces the Python/JAX implementations exactly: the annual rate is
# gathered and converted to a monthly rate in the projection, for every model
# point in every month. `:table` gathers a pre-converted monthly rate instead.
# Both are singletons wrapped in `Val` so they ride through a broadcast as
# scalars and specialise the generated kernel.

@inline mort_rate_mth(::Val{:inline}, mort, row, col) =
    1 - (1 - (@inbounds mort[row, col]))^(1 / 12)
@inline mort_rate_mth(::Val{:table}, mort, row, col) = @inbounds mort[row, col]

@inline function lapse_rate_mth(::Val{:inline}, lapse, dur)
    rate = max(0.02, 0.1 - 0.02 * dur)
    return 1 - (1 - rate)^(1 / 12)
end
@inline lapse_rate_mth(::Val{:table}, lapse, dur) =
    @inbounds lapse[clamp(dur, Int32(0), Int32(4)) + 1]

mort_table(assume::Assumptions, ::Val{:inline}) = assume.mort_ann
mort_table(assume::Assumptions, ::Val{:table}) = assume.mort_mth

# --- reporting -------------------------------------------------------------
commas(n::Integer) = replace(string(n), r"(?<=[0-9])(?=(?:[0-9]{3})+$)" => ",")

# Sum of discounted net cash flow for one copy of lifelib's 10,000 model points,
# as produced by the reference Python implementations. The result is linear in the
# multiplier, so this doubles as a correctness check at any portfolio size.
const REFERENCE_RESULT = 215146132.0684811

function report(label, n, result, seconds, multiplier)
    expected = REFERENCE_RESULT * multiplier
    println(label)
    println("number modelpoints=", commas(n))
    println("result=", result)
    println("relative_error=", (result - expected) / expected)
    println("time_in_seconds=", seconds)
    return nothing
end

"""
    time_run(f)

Time `f()`, using the backend's timer if it installed one (CUDA events, so the
measurement covers asynchronous work) and the monotonic clock otherwise.
"""
function time_run(f)
    timer = TIME_RUN[]
    timer === nothing || return timer(f)
    t0 = time_ns()
    result = f()
    return result, (time_ns() - t0) / 1e9
end
