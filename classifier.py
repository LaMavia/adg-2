from collections import Counter, defaultdict, deque
import math
import sys
from typing import Iterator
import pandas as pd
from Bio import SeqIO 
import gzip
import mmh3
from tqdm import tqdm
from array import array
import itertools
import os
from random import Random

MAX_U32 = 2**32 - 1
CSV_DELIMITER = "\t"

Sketch = array[int]
Class = list[Sketch]

class Bit(object):
    TYPE = 'B'
    TYPE_LEN = 8

    def __init__(self, size: int):
        self.bits = array(Bit.TYPE, [0]*((size+Bit.TYPE_LEN-1)//Bit.TYPE_LEN))

    def set(self, bit: int):
        b = self.bits[bit//Bit.TYPE_LEN]
        self.bits[bit//Bit.TYPE_LEN] = b | 1 << (bit % Bit.TYPE_LEN)

    def __getitem__(self, bit: int):
        b = self.bits[bit//Bit.TYPE_LEN]
        return (b >> (bit % Bit.TYPE_LEN)) & 1


class BloomFilter(object):
    def __init__(self, m: int, k: int):
        self.m = m
        self.k = k
        self.bits = Bit(m)
        self.rand = Random()
        self.hashes = [
            lambda v: mmh3.hash(v, seed=i, signed=False)
            for i in range(k)
        ]

    def __contains__(self, key: str):
        for i in self._indexes(key): 
            if not self.bits[i]:
                return False    
        return True 

    def add(self, key: str):
        for i in self._indexes(key): 
            self.bits.set(i)

    def _indexes(self, key: str):
        return (h(key) % self.m for h in self.hashes)

def batch_counter(iter: Iterator[str], w: int) -> Iterator[tuple[str, int]]:
    for batch in itertools.batched(iter, w):
        counter = Counter(batch)
        yield from counter.items()

def csv_length(path: str) -> int:
    with open(path) as f:
        return deque(enumerate(f), 1).pop()[0]

def make_seq_gen(path: str) -> Iterator[str]:
    open_fun = gzip.open if path.endswith('.gz') else open

    with open_fun(path, 'rt') as handle:
        yield from (str(record.seq) for record in SeqIO.parse(handle, "fasta"))

def running_set[T](max_size: int, iter: Iterator[T]) -> Iterator[T]:
    for elems in itertools.batched(iter, max_size):
        yield from set(elems)

def training_sets(path: str) -> Iterator[tuple[str, Iterator[str]]]:
    root = os.path.dirname(os.path.realpath(path))
    datasets = pd.read_csv(path, delimiter=CSV_DELIMITER)
    class_paths = defaultdict(list)
    for _, (set_path, class_label, *_) in datasets.iterrows():
        class_paths[class_label].append(f'{root}/{set_path}')

    for class_label, data_paths in class_paths.items():
        yield class_label, itertools.chain(*map(make_seq_gen, data_paths))

def test_sets(path: str) -> Iterator[tuple[str, Iterator[str]]]:
    root = os.path.dirname(os.path.realpath(path))
    datasets = pd.read_csv(path, delimiter=CSV_DELIMITER)
    for _, (set_path, *_) in datasets.iterrows():
        open_fun = gzip.open if set_path.endswith('.gz') else open

        with open_fun(f'{root}/{set_path}', 'rt') as handle:
            yield set_path, (str(record.seq) for record in SeqIO.parse(handle, "fasta"))

def canonical_kmer(kmer: str) -> str:
    complement = str.maketrans("ACGT", "TGCA")
    rc = kmer.translate(complement)[::-1]
    return min(kmer, rc)

def kmer_generator(sequences: Iterator[str], k: int) -> Iterator[str]:
    for seq in sequences:
        n = len(seq)
        for i in range(n - k + 1):
            kmer = seq[i:i+k]
            if 'N' in kmer:
                continue

            yield canonical_kmer(kmer)

def minhash_similarity(sketch_size: int, a: Sketch, b: Sketch) -> float:
    return sum(a[i] == b[i] for i in range(sketch_size)) / sketch_size

def classify_sketch(sketch_size: int, classes: dict[str, Sketch], sketch: Sketch) -> dict[str, float]:
    classification = {}
    for class_name, class_sketch in classes.items():
        classification[class_name] = minhash_similarity(sketch_size, class_sketch, sketch)
            
    return classification

def compute_minhash(kmers: Iterator[str], m: int) -> Sketch:
    """
    Compute a MinHash sketch of size m from an iterator of k-mers.
    Sketch[i] = minimum hash observed under seed i
    """

    # N_FILTERS = 3
    # FILTER_LEN = 128
    # FILTER_HASHES = 10
    # occurence_filters = [
    #     BloomFilter(FILTER_LEN, FILTER_HASHES)
    #     for _ in range(N_FILTERS)
    # ]
    W = 2_000_000
    C = optimal_k(W)
    sketch = array('L', [MAX_U32] * m)

    for kmer, cnt in batch_counter(kmers, 100_000):
        if cnt < C:
            continue
        for i in range(m):
            h = mmh3.hash(kmer, seed=i, signed=False)
            if h < sketch[i]:
                sketch[i] = h

    # for kmer in kmers:
    #     for i in range(m):
    #         h = mmh3.hash(kmer, seed=i, signed=False)
    #         if h < sketch[i]:
    #             sketch[i] = h

    return sketch

def optimal_k(n: int) -> int:
    q = 0.01
    return math.ceil(math.log(n * (1 - q) / q, 4))

def main():
    # python3 classifier.py training_data.tsv testing_data.tsv output.tsv
    _, training_path, test_path, output_path = sys.argv

    k = 14
    m = 256

    sample_sketches: dict[str, Sketch] = {}

    bar = tqdm(training_sets(training_path), desc="Computing training sketches")
    for cls_label, sequences in bar:
        bar.set_postfix_str(cls_label)
        kmers = kmer_generator(sequences, k=k)
        sketch = compute_minhash(kmers, m=m)
        sample_sketches[cls_label] = sketch

    out_data = defaultdict(list)
    for path, sequences in tqdm(test_sets(test_path), total=csv_length(test_path), desc="Comparing test reads"):
        kmers = kmer_generator(sequences, k=k)
        sketch = compute_minhash(kmers, m=m)

        classification = classify_sketch(m, sample_sketches, sketch)
        out_data['path'].append(path)
        for cls_label, probability in classification.items():
            out_data[cls_label].append(f'{probability:.03f}')

    pd.DataFrame(out_data).to_csv(output_path, sep=CSV_DELIMITER, header=True, index=False)

if __name__ == "__main__":
    main()
    

