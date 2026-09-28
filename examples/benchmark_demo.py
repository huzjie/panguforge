import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Run the benchmark suite."""
import json
from panguforge.config import load_config
from panguforge.benchmark.benchmarks import run_bench


def main():
    cfg = load_config("config.example.yaml")
    print(json.dumps(run_bench(cfg), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
