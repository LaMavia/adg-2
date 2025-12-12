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

MAX_U32 = 2**32 - 1
CSV_DELIMITER = "\t"

Sketch = array[int]
Class = list[Sketch]


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
    datasets = pd.read_csv(path, delimiter=CSV_DELIMITER)
    class_paths = defaultdict(list)
    for _, (set_path, class_label, *_) in datasets.iterrows():
        class_paths[class_label].append(set_path)

    for class_label, data_paths in class_paths.items():
        yield class_label, itertools.chain(*map(make_seq_gen, data_paths))

def test_sets(path: str) -> Iterator[tuple[str, Iterator[str]]]:
    datasets = pd.read_csv(path, delimiter=CSV_DELIMITER)
    for _, (dpath, *_) in datasets.iterrows():
        open_fun = gzip.open if dpath.endswith('.gz') else open

        with open_fun(dpath, 'rt') as handle:
            yield dpath, (str(record.seq) for record in SeqIO.parse(handle, "fasta"))

def canonical_kmer(kmer: str) -> str:
    complement = str.maketrans("ACGT", "TGCA")
    rc = kmer.translate(complement)[::-1]
    return min(kmer, rc)

def kmer_generator(sequences: Iterator[str], k: int) -> Iterator[str]:
    for seq in sequences:
        n = len(seq)
        for i in range(n - k + 1):
            kmer = seq[i:i+k]
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
    C = 4
    sketch = array('L', [MAX_U32] * m)
    candidate_set = Counter(kmers)
    for kmer, cnt in candidate_set.items():
        if cnt < C:
            continue
        for i in range(m):
            h = mmh3.hash(kmer, seed=i, signed=False)
            if h < sketch[i]:
                sketch[i] = h

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
    

