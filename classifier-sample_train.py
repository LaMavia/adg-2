from collections import Counter, defaultdict, deque
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
import random
import multiprocessing as mp

MAX_U32 = 2**32 - 1
CSV_DELIMITER = "\t"


Sketch = array[int]
Class = list[Sketch]

def main(output_path: str, RATE: int):

    def batch_counter(iter: Iterator[str], w: int) -> Iterator[tuple[str, int]]:
        for batch in itertools.batched(iter, w):
            counter = Counter(batch)
            yield from counter.items()

    def csv_length(path: str) -> int:
        with open(path) as f:
            return deque(enumerate(f), 1).pop()[0]

    def make_seq_gen(path: str, sample: bool) -> Iterator[str]:
        open_fun = gzip.open if path.endswith('.gz') else open

        with open_fun(path, 'rt') as handle:
            if sample:
                yield from (str(record.seq) for record in SeqIO.parse(handle, "fasta") if random.random() <= RATE / 100)
            else:
                yield from (str(record.seq) for record in SeqIO.parse(handle, "fasta"))


    def training_sets(path: str) -> Iterator[tuple[str, list[Iterator[str]]]]:
        root = os.path.dirname(os.path.realpath(path))
        datasets = pd.read_csv(path, delimiter=CSV_DELIMITER)
        class_paths = defaultdict(list)
        for _, (set_path, class_label, *_) in datasets.iterrows():
            class_paths[class_label].append(f'{root}/{set_path}')

        for class_label, data_paths in class_paths.items():
            yield class_label, list([make_seq_gen(dp, True) for dp in data_paths])

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

    def compute_minhash(kmers: Iterator[str], m: int, c: int) -> Sketch:
        """
        Compute a MinHash sketch of size m from an iterator of k-mers.
        Sketch[i] = minimum hash observed under seed i
        """

        W = 2_000_000
        sketch = array('L', [MAX_U32] * m)

        for kmer, cnt in batch_counter(kmers, W):
            if cnt < c:
                continue
            for i in range(m):
                h = mmh3.hash(kmer, seed=i, signed=False)
                if h < sketch[i]:
                    sketch[i] = h

        return sketch


    # python3 classifier.py training_data.tsv testing_data.tsv output.tsv
    _, training_path, test_path = sys.argv
    i = mp.current_process()._identity[0]

    k = 14
    m = 256

    len_list = []
    for cls_label, sequences in training_sets(training_path):
        l = sum(sum(1 for _ in s) / len(sequences) for s in sequences)
        len_list.append(l)

    avg_len = sum(len_list) / len(len_list)
    a = 10 / (1e6 - 1e3)
    b = 4 - 1000 * a
    c = round(a * avg_len + b)

    sample_sketches: dict[str, Sketch] = {}

    bar = tqdm(training_sets(training_path), desc=f"[{output_path}] Computing training sketches", position=i, leave=False)
    for cls_label, sequences in bar:
        bar.set_postfix_str(cls_label)
        kmers = kmer_generator(itertools.chain(*sequences), k=k)
        sketch = compute_minhash(kmers, m=m, c=c)
        sample_sketches[cls_label] = sketch

    bar.close()
    out_data = defaultdict(list)
    bar = tqdm(test_sets(test_path), total=csv_length(test_path), desc=f"[{output_path}] Comparing test reads", position=i, leave=False)
    for path, sequences in bar:
        kmers = kmer_generator(sequences, k=k)
        sketch = compute_minhash(kmers, m=m, c=c)

        classification = classify_sketch(m, sample_sketches, sketch)
        out_data['path'].append(path)
        for cls_label, probability in classification.items():
            out_data[cls_label].append(f'{probability:.03f}')

    bar.close()
    pd.DataFrame(out_data).to_csv(output_path, sep=CSV_DELIMITER, header=True, index=False)

if __name__ == "__main__":
    rates = [(i, r) for i in range(10) for r in range(10, 101, 10)]
    p = mp.Pool(10)

    with p:
        p.starmap(main, [(f'./output-{r:03d}-{i:02d}.tsv', r) for i, r in rates])

    

