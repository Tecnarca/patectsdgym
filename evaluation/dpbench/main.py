import argparse

from dpbench import benchmark
from dpbench import conf
import mlflow

def _parse_args():
    parser = argparse.ArgumentParser(prog="dpBench", 
                                    description="Differentially private synthetic data generators benchmarking suite", 
                                    epilog="Sample command: python main.py -d bank adult -e 0.01 0.1 1 -m pmse ml_eval wasserstein rsa -r test_run")

    parser.add_argument('-d', '--dataset', nargs="+", default=conf.KNOWN_DATASETS, help="Datasets names on which the benchmarks will be executed")
    parser.add_argument('-e', '--epsilon', nargs="+", default=conf.EPSILONS, help="Epsilons values for which the models will be evaluated")
    parser.add_argument('-m', '--metric', nargs="+", default=conf.KNOWN_METRICS, help="Differential privacy metrics for which the models will be evaluated")
    parser.add_argument('-r', '--run', default="dpbench_run", help="Run name for mlflow")
    parser.add_argument('-p', '--path', default="results/", help="Path of the folder to save the results")

    return parser.parse_args()

if __name__ == "__main__":

    args = _parse_args()

    benchmark.run(epsilons=args.epsilon, run_name=args.run, metric=args.metric, dataset=args.dataset, result_path=args.path)        