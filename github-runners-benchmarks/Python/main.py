from mortality import run_mortality_benchmarks
from basicterm_m import run_basic_term_benchmarks
from savings_me import run_savings_benchmarks
from basicterm_me import run_basic_term_me_benchmarks
import yaml


def get_results():
    return {
        "mortality": run_mortality_benchmarks(),
        "basic_term_benchmark": run_basic_term_benchmarks(),
        "basic_term_me_benchmark": run_basic_term_me_benchmarks(),
        "savings_benchmark": run_savings_benchmarks(),
    }


if __name__ == "__main__":
    results = get_results()
    # write to benchmark_results.yaml
    with open("benchmark_results.yaml", "w") as f:
        yaml.dump(results, f)
