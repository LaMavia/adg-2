import sys
import csv
import pandas as pd
from typing import Iterator

def training_sets(path: str):
    datasets = pd.read_csv(path, delimiter="\t")
    for data_path, data_class in datasets:
        print(f"{data_path=}, {data_class=}")


def main():
    # python3 classifier.py training_data.tsv testing_data.tsv output.tsv
    _, training_path, testinng_path, output_path = sys.argv

    training_sets(training_path)
    

