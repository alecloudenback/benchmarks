# Benchmarking

Benchmarks in this repository:

* `basic_term_benchmark`: Replicate the cashflows of the [LifeLib BasicTerm model](https://github.com/lifelib-dev/lifelib/tree/main/lifelib/libraries/basiclife/BasicTerm_M)
    * Python [LifeLib BasicTerm_M](https://github.com/lifelib-dev/lifelib/tree/main/lifelib/libraries/basiclife/BasicTerm_M)
    * Julia [using memoization](https://github.com/actuarialopensource/benchmarks/blob/main/Julia/src/Benchmarks.jl)
    * Python [using custom memoization decorator](https://github.com/actuarialopensource/benchmarks/blob/main/Python/basicterm_scratch.py)
    * Python [using JAX](https://github.com/actuarialopensource/benchmarks/blob/main/Python/basicterm_jax.py)
* `exposures`: Create date partitions for experience studies
    * Julia [ExperienceAnalysis](https://github.com/JuliaActuary/ExperienceAnalysis.jl)
    * R [actxps](https://github.com/mattheaphy/actxps)
* `mortality`: Read SOA mortality tables and use them in a simple calculation
    * Julia [MortalityTables](https://github.com/JuliaActuary/MortalityTables.jl)
    * Python [Pymort](https://github.com/actuarialopensource/pymort)

The below results are generated by the benchmarking scripts in the folders for each language. These scripts are run automatically by GitHub Actions and populate the results below. 

```yaml 
basic_term_benchmark:
- Julia Benchmarks basic_term:
    mean: TrialEstimate(192.127 ms)
    result: 1.4489630534602132e7
- Python jax basic_term_m:
    mean: 337.39129650000166 milliseconds
    result: 14489630.53460337
  Python lifelib basic_term_m:
    mean: 1182.7541804499986 milliseconds
    result: 14489630.534601536
  Python scratch basic_term_m:
    mean: 957.6274868500008 milliseconds
    result: 14489630.534603368
exposures:
- Julia ExperienceAnalysis.jl:
    mean: TrialEstimate(40.691 ms)
    num_rows: 141281
- R actxps:
    mean: 887.451163 ms
    num_rows: 141281
mortality:
- Julia MortalityTables.jl:
    mean: TrialEstimate(411.124 μs)
    result: 1904.4865526636793
- Python PyMort:
    mean: 2490.7538589499995 milliseconds
    result: 1904.4865526636793
```
