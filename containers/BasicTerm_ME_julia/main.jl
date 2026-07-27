include(joinpath(@__DIR__, "common.jl"))

const USAGE = """
usage: main.jl [--multiplier N] [--model array|kernel|reactant]
               [--rates inline|table] [--runs N] [--cpu]

  --multiplier N       repeat the 10,000 lifelib model points N times (default: 100)
  --model MODEL        array    : time-stepped CUDA.jl array model
                       kernel   : hand-written CUDA kernel (default)
                       reactant : traced to StableHLO and compiled by XLA
  --rates MODE         inline : convert annual rates to monthly inside the
                                projection, as the Python/JAX models do
                       table  : look up pre-converted monthly rates
                       (default: table for kernel, inline for the others)
  --runs N             number of timed runs, the fastest is reported (default: 1)
  --cpu                force the CPU backend (for validation without a GPU)

Only the backend actually selected is loaded, so a reactant run never loads
CUDA.jl and vice versa.
"""

const MODELS = ("array", "kernel", "reactant")
const STRING_OPTS = ("model", "rates")

function parse_args(argv)
    opts = Dict{String, Any}("multiplier" => 100, "model" => "kernel", "runs" => 1,
                             "rates" => nothing, "cpu" => false)
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--cpu"
            opts["cpu"] = true
            i += 1
        elseif arg in ("-h", "--help")
            print(USAGE)
            exit(0)
        elseif startswith(arg, "--") && i < length(argv)
            key = arg[3:end]
            haskey(opts, key) || error("unknown option $arg\n\n$USAGE")
            opts[key] = key in STRING_OPTS ? argv[i + 1] : parse(Int, argv[i + 1])
            i += 2
        else
            error("could not parse $arg\n\n$USAGE")
        end
    end
    opts["model"] in MODELS || error("--model must be one of $(join(MODELS, ", "))")
    if opts["rates"] === nothing
        opts["rates"] = opts["model"] == "kernel" ? "table" : "inline"
    end
    opts["rates"] in ("inline", "table") || error("--rates must be inline or table")
    return opts
end

const BACKEND_FILE = Dict("array" => "term_me_array_cuda.jl",
                          "kernel" => "term_me_kernel_cuda.jl",
                          "reactant" => "term_me_reactant.jl")

function main(argv = ARGS)
    opts = parse_args(argv)

    # Loaded here rather than at the top of the file so that only the requested
    # backend's package is brought into the process.
    include(joinpath(@__DIR__, BACKEND_FILE[opts["model"]]))
    Base.invokelatest(ENTRYPOINTS[:init], opts["cpu"])
    Base.invokelatest(ENTRYPOINTS[:run], opts["multiplier"], opts["runs"],
                      Symbol(opts["rates"]))
    return nothing
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
