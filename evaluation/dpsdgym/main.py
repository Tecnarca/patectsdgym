import sys
import os
import time
import mlflow
import json
import argparse
import textwrap

import conf

from load_data import load_data

from synthesis import run_all_synthesizers
from evaluate import run_ml_eval, run_wasserstein, run_pMSE, run_sra

from warnings import simplefilter

def run(epsilons, run_name, flags, dataset):
    loaded_datasets = load_data(dataset)
    data_dicts = run_all_synthesizers(loaded_datasets, epsilons)
    
    if 'wasserstein' in flags:
        run_wasserstein(data_dicts, 50, run_name)
    if 'pmse' in flags:
        run_pMSE(data_dicts, run_name)
    if 'ml_eval' in flags:
        results = run_ml_eval(data_dicts, epsilons, run_name)
        if 'sra' in flags:
            results = run_sra(results)
        print(results)

    with open('artifact.json', 'w') as f:
        json.dump(results, f)
    mlflow.log_artifact("artifact.json", "results.json")

def _parse_args():
    parser = argparse.ArgumentParser(prog="DPSDGYM", 
                                    description="Differentially private synthetic data generators evaluation suite", 
                                    epilog="Sample command: python main.py -d bank adult -e 0.01 0.1 1 -m pmse ml_eval wasserstein rsa -r test_run")

    parser.add_argument('-d', '--dataset', nargs="+", default=conf.KNOWN_DATASETS, help="Datasets names on which the benchmarks will be executed")
    parser.add_argument('-e', '--epsilon', nargs="+", default=conf.EPSILONS, help="Epsilons values for which the models will be evaluated")
    parser.add_argument('-m', '--metric', nargs="+", default=conf.KNOWN_METRICS, help="Differential privacy metrics for which the models will be evaluated")
    parser.add_argument('-r', '--run', default="test_run", help="Run name for mlflow")

    return parser.parse_args()

if __name__ == "__main__":

    args = _parse_args()

    with mlflow.start_run(run_name=args.run):
        mlflow.log_param("epsilons", str(args.epsilon))
        mlflow.log_param("dataset", args.dataset)
        mlflow.log_param("flags", str(args.metric))
        run(epsilons=args.epsilon, run_name=args.run, flags=args.metric, dataset=args.dataset)
