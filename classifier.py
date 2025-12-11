import sys
from typing import Iterator
import pandas as pd
from Bio import SeqIO 
import gzip
import mmh3
import numpy as np


def training_sets(path: str) -> Iterator[tuple[str, Iterator[str]]]:
    datasets = pd.read_csv(path, delimiter="\t")
    for _, (dpath, dcls, *_) in datasets.iterrows():
        open_fun = gzip.open if dpath.endswith('.gz') else open

        with open_fun(dpath, 'rt') as handle:
            yield dcls, (str(record.seq).upper() for record in SeqIO.parse(handle, "fasta"))

def test_sets(path: str) -> Iterator[Iterator[str]]:
    datasets = pd.read_csv(path, delimiter="\t")
    for _, (dpath, *_) in datasets.iterrows():
        open_fun = gzip.open if dpath.endswith('.gz') else open

        with open_fun(dpath, 'rt') as handle:
            yield (str(record.seq).upper() for record in SeqIO.parse(handle, "fasta"))

def canonical_kmer(kmer: str) -> str:
    complement = str.maketrans("ACGT", "TGCA")
    rc = kmer.translate(complement)[::-1]
    return min(kmer, rc)

def kmer_generator(sequences: Iterator[str], k: int) -> Iterator[str]:
    for seq in sequences:
        n = len(seq)
        if n < k:
            continue
        for i in range(n - k + 1):
            kmer = seq[i:i+k]
            if 'N' in kmer:
                continue
            yield canonical_kmer(kmer)

def compute_minhash(kmers: Iterator[str], m: int = 256) -> np.ndarray:
    """
    Compute a MinHash sketch of size m from an iterator of k-mers.
    Sketch[i] = minimum hash observed under seed i
    """
    sketch = np.full(m, np.iinfo(np.uint64).max, dtype=np.uint64)
    for kmer in kmers:
        for i in range(m):
            h = mmh3.hash(kmer, seed=i, signed=False)
            if h < sketch[i]:
                sketch[i] = h
    return sketch

def main():
    # python3 classifier.py training_data.tsv testing_data.tsv output.tsv
    _, training_path, testing_path, output_path = sys.argv

    k = 21
    m = 256

    sample_sketches = {}

    for cls_label, sequences in training_sets(training_path):
        print(f"Processing training class: {cls_label}")
        kmers = kmer_generator(sequences, k=k)
        sketch = compute_minhash(kmers, m=m)
        sample_sketches[cls_label] = sketch
        print(f"  MinHash sketch computed for class {cls_label}, size={m}")
    print(f"\nComputed MinHash sketches for {len(sample_sketches)} training datasets.")

    for cls_label, sketch in sample_sketches.items():
        print(f"Class: {cls_label}, Sketch:\n{sketch}\n")

if __name__ == "__main__":
    main()
    

