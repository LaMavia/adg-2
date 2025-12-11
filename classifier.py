import sys
from typing import Iterator
import pandas as pd
from Bio import SeqIO 
import gzip

def training_sets(path: str) -> Iterator[tuple[str, Iterator[str]]]:
    datasets = pd.read_csv(path, delimiter="\t")
    for _, (dpath, dcls, *_) in datasets.iterrows():
        open_fun = gzip.open if dpath.endswith('.gz') else open

        with open_fun(dpath, 'rt') as handle:
            yield dcls, (str(record['seq']) for record in SeqIO.parse(handle, "fasta"))

def test_sets(path: str) -> Iterator[Iterator[str]]:
    datasets = pd.read_csv(path, delimiter="\t")
    for _, (dpath, *_) in datasets.iterrows():
        open_fun = gzip.open if dpath.endswith('.gz') else open

        with open_fun(dpath, 'rt') as handle:
            yield (str(record['seq']) for record in SeqIO.parse(handle, "fasta"))

def main():
    # python3 classifier.py training_data.tsv testing_data.tsv output.tsv
    _, training_path, testinng_path, output_path = sys.argv

    training_sets(training_path)

if __name__ == "__main__":
    main()
    

