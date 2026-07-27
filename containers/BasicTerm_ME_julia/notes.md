# BasicTerm_ME, Julia (CUDA.jl and Reactant.jl)

Julia port of the `BasicTerm_ME_python` container. Same model, same input
workbooks, same 100M model point workload; three implementations that make
different tradeoffs.

Only the backend you select is loaded, so a `--model reactant` run never brings
CUDA.jl into the process and a `--model array|kernel` run never brings in
Reactant. That keeps two independent GPU runtimes from initialising side by
side, at the cost of a fairly large image carrying both.

## Running

```bash
docker build . -t basicterm_me_julia
docker run --gpus all basicterm_me_julia                                # kernel, 100M
docker run --gpus all basicterm_me_julia --model array --multiplier 10000
docker run --gpus all basicterm_me_julia --model reactant --multiplier 10000
docker run basicterm_me_julia --cpu --multiplier 1                      # no GPU, validation
```

To run every implementation in this repo side by side, use
`containers/run_all.sh` one directory up.

Without Docker:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. --threads=auto main.jl --model kernel --multiplier 10000
```

The image must be built for the architecture it runs on (`--platform
linux/amd64` for the A100/H100 hosts in the top-level README).

## The three implementations

### `term_me_array_cuda.jl` — array model

The analogue of `term_me_iterative_jax.py`, and written to be directly
comparable to it: the same arithmetic in the same order, including converting
annual mortality and lapse rates to monthly ones over the whole portfolio in
every one of the 277 months. State is three length-`n` vectors, and each month
issues three fused elementwise kernels plus one fused reduction. The whole
portfolio moves in lockstep, so every model point is touched in every month
whether or not it is in force, and the policy state round-trips through global
memory 277 times. That memory traffic — roughly
`277 x n x 8 bytes x (arrays touched)` — is what sets the floor on this design.

Two Julia-specific details worth noting:

* Lookup tables are passed through broadcasts as `Ref(table)`. Broadcasting
  treats a `Ref` as a scalar, and CUDA.jl adapts `Ref{CuArray}` into a
  device-side reference, so the mortality/lapse gathers stay inside the fused
  kernel instead of materialising an intermediate.
* The net cash flow is built as an un-materialised `Broadcasted` and handed
  straight to `mapreduce`, so the monthly total costs one reduction kernel and
  no length-`n` temporary.

### `term_me_kernel_cuda.jl` — kernel model

One thread per model point, walking that policy through its own lifetime. The
optimisations, roughly in order of how much they matter:

1. **Policy state stays in registers.** In force / deaths / lapses never reach
   global memory. The model point data is read once for the whole projection
   instead of once per month, which removes the array model's dominant cost.
2. **Each policy only runs its own months.** The loop runs from
   `max(0, -duration_mth)` (new business written mid-projection) to maturity,
   not over all 277 months of the longest policy in the portfolio.
3. **Assumption lookups hoisted to the anniversary.** `mort_rate` and
   `lapse_rate` only change when `duration` changes, so the month loop is nested
   inside a policy-year loop and the lookups happen ~1/12 as often.
4. **Monthly rates precomputed on the host** (`rates=table`, the default here).
   Annual-to-monthly conversion is a function of small tables only — 103x6
   mortality cells and 5 distinct lapse rates — so it is done once at setup
   rather than as a `pow` over the portfolio in the projection.
5. **Discount and inflation staged into shared memory.** They are the same 277
   values for every thread in the block.
6. **Grid-stride over an occupancy-sized grid.** Per-policy loop lengths vary a
   lot (10-, 15- and 20-year terms at every duration), so each thread taking
   many policies averages the divergence out across the grid.

A further optimisation is deliberately *not* taken: the benchmark builds its
100M model points by tiling 10,000 real ones with `np.tile` semantics
(`ABCABC...`), so neighbouring threads get unrelated policies. Tiling
`AAA...BBB...` instead, or sorting the portfolio by remaining term, would make
each warp coherent and cut the divergence — but that is exploiting an artifact
of how the benchmark manufactures its workload rather than a property of a real
portfolio, so the ordering is left matching the Python containers.

### `term_me_reactant.jl` — Reactant / XLA model

Ordinary Julia array code traced to StableHLO and compiled by XLA, the same
backend the JAX implementation uses. `@trace while` lowers the 277-month loop
into a StableHLO `while` — the analogue of `lax.scan`, rather than unrolling 277
copies of the body — and XLA fuses the whole program.

This row exists to separate two questions the table otherwise conflates: how
fast XLA is on this model, and how fast the Python/JAX frontend is. Same model,
same compiler, different language.

Two tracing details cost real time to find, and both produce plausible numbers
rather than errors:

* **Scalars carried across a `@trace while` must already be traced.** Writing
  `total = 0.0` before the loop makes it a constant: the loop still runs, and
  the accumulated value is silently discarded. You get `0.0` back, not an error.
  `promote_to_traced` is what makes it a loop carry.
* **A plain `Bool` referenced in the loop body is lifted into a
  `TracedRNumber{Bool}`**, which then cannot be used as an `if` condition. The
  rate mode is therefore resolved by dispatch on a `Val` argument, which is not
  traceable and survives as a compile-time constant.

Also worth knowing: `fld` on traced integers currently lowers to *truncating*
division. Here that is immaterial — it only differs when `duration_mth + t < 0`,
i.e. new business not yet written, where policies in force are zero and every
affected term is multiplied by that zero — but it is the kind of thing that
would quietly change answers in another model. The reference check is what
confirms it.

## The `--rates` switch

`--rates inline|table` selects whether annual-to-monthly rate conversion happens
in the projection or once at setup. It is model-equivalent — all six
combinations produce the same total — so it isolates one optimisation:

| model    | `--rates` | why |
|----------|-----------|-----|
| array    | `inline` (default) | matches the JAX scan statement for statement, so the array row is directly comparable to the JAX row |
| array    | `table` | same array design, without the per-month `pow` over the portfolio |
| kernel   | `table` (default) | the optimised implementation |
| kernel   | `inline` | the kernel's cost with the conversion put back (still hoisted to the policy year, so this is not a like-for-like ablation against the array model) |
| reactant | `inline` (default) | matches JAX; this is the like-for-like XLA comparison |
| reactant | `table` | whether the JAX row could be improved the same way |

Run `array inline` vs `array table` to see how much of the kernel's advantage is
rate precomputation rather than the change in memory layout.

## Other deviations from the Python implementations

* **Integer model point fields are `Int32`** in the CUDA models and `Int64` in
  the Reactant model, which matches the JAX implementation running under
  `jax_enable_x64`. Floating point is `Float64` everywhere.
* Everything else — the 277-month projection length, the discount and inflation
  conventions, expense/commission/lapse rules and the input workbooks — is
  identical.

## Correctness

All three implementations reproduce the reference total, one copy of the model
point table summing to `215,146,132.0684811` and scaling linearly with the
multiplier.
Each run prints `relative_error` against that reference; expect ~1e-13, which is
floating point summation order, not a model difference.

The kernel sums each policy over time and then across policies, while the array
and Reactant models sum across policies and then over time, so they agree to
about 1e-13 rather than bit-for-bit.

## CPU backend

`--cpu` runs the array and kernel models on `Array` instead of `CuArray` (the
kernel model's per-policy function is shared verbatim with a `Threads.@threads`
driver) and the Reactant model on XLA's CPU backend. This is for validating the
models without a GPU — the benchmark numbers in the top-level README are the GPU
path.
