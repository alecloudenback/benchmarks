# Both CUDA.jl and Reactant.jl ship their own GPU runtimes as artifacts, so only
# the host driver (supplied by `docker run --gpus all`) is needed - no CUDA base
# image required. That does make this a large image: it carries CUDA.jl's toolkit
# and Reactant's CUDA-enabled XLA build side by side.
#
# Build for the architecture you will run on, e.g.
#   docker build --platform linux/amd64 . -t basicterm_me_julia
ARG JULIA_VERSION=1.11
FROM julia:${JULIA_VERSION}-bookworm

# CUDA toolkit for CUDA.jl, and the matching XLA build for Reactant. These are
# separate runtimes with independently versioned artifacts; only one is ever
# loaded in a given run.
ARG CUDA_VERSION=12.6
ARG REACTANT_CUDA_VERSION=12.9

# Trailing `:` keeps the bundled (precompiled stdlib) depot on the path.
ENV JULIA_DEPOT_PATH=/usr/local/share/julia: \
    JULIA_PROJECT=/app \
    JULIA_CPU_TARGET=generic

WORKDIR /app
COPY Project.toml /app/

# Pin both GPU runtimes so their artifacts are baked into the image instead of
# being resolved on the first run. Reactant_jll's `gpu`/`gpu_version` are
# compile-time preferences and it otherwise auto-detects from the local driver -
# which would silently select the CPU-only build here, since the build host has
# no GPU. They must therefore be set before instantiating.
RUN printf '[CUDA_Runtime_jll]\nversion = "%s"\n\n[Reactant_jll]\ngpu = "cuda"\ngpu_version = "%s"\n' \
        "${CUDA_VERSION}" "${REACTANT_CUDA_VERSION}" > /app/LocalPreferences.toml \
    && julia -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' \
    && julia -e 'using CUDA; CUDA.precompile_runtime()'

COPY . /app/

# Smoke test: run every model/rate combination on the CPU backend, each of which
# checks its result against the reference total. One process per combination,
# since a run only ever loads its own backend. Device code is still compiled on
# first launch, which is why every model does a warm-up run before timing.
RUN for model in array kernel reactant; do \
        for rates in inline table; do \
            julia /app/main.jl --cpu --multiplier 1 --model "$model" --rates "$rates" \
                || exit 1; \
        done; \
    done

ENTRYPOINT ["julia", "--project=/app", "--threads=auto", "/app/main.jl"]
CMD ["--multiplier", "10000", "--model", "kernel"]
