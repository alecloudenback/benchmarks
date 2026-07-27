using CUDA

@isdefined(NSTEPS) || include(joinpath(@__DIR__, "common.jl"))

# CUDA backend for the array and kernel models. Both are written against
# `AbstractArray`, so the same code runs on `CuArray` (GPU) or `Array` (CPU).
# The CPU path exists so the models can be validated without a GPU; the
# benchmark numbers come from the GPU.

on_gpu() = ADAPT[] === CuArray

function init_backend!(force_cpu::Bool)
    if !force_cpu && CUDA.functional()
        ADAPT[] = CuArray
        # CUDA events, so the measurement covers the asynchronous work.
        TIME_RUN[] = function (f)
            result = Ref{Any}()
            seconds = CUDA.@elapsed result[] = f()
            return result[], seconds
        end
        println("device=", CUDA.name(CUDA.device()))
    else
        println("device=cpu (CUDA.functional()=", CUDA.functional(),
                ", threads=", Threads.nthreads(), ")")
    end
    return nothing
end

ENTRYPOINTS[:init] = init_backend!
