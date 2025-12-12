#!/usr/bin/env python3
from Bio import SeqIO
from sys import argv
import hashlib
from collections import defaultdict, deque
from array import array
import math


def get_minimizers(seq: str, k: int, w: int):
    window = deque()
    for i, h in enumerate(rolling_kmer_hashes(seq, k)):
        window.append(h)
        if len(window) > w:
            window.popleft()
        if len(window) == w:
            yield i - w + 1, min(window)

def choose_k_for_error(err_rate: float):
    """Pick k such that (1-err_rate)^k ~ 0.25-0.3 to tolerate 5-10% errors."""
    k = max(10, min(20, int(-math.log(0.25)/math.log(1-err_rate))))
    return k

class MashMapIndex:
    def __init__(self, ref_seq: str, k: int, windows=[15,30,60]):
        self.k = k
        self.windows = windows
        self.index = defaultdict(lambda: array('I'))
        self.build_index(ref_seq)

    def build_index(self, ref_seq: str):
        for w in self.windows:
            for pos, m in get_minimizers(ref_seq, k=self.k, w=w):
                self.index[(w,m)].append(pos)

    def query(self, read_seq: str):
        hits = defaultdict(int)
        for w in self.windows:
            for read_pos, m in get_minimizers(read_seq, k=self.k, w=w):
                ref_positions = self.index.get((w,m))
                if ref_positions:
                    for ref_pos in ref_positions:
                        hits[ref_pos - read_pos] += 1
        return hits

def canonical_kmer(kmer: str) -> str:
    complement = str.maketrans("ACGT", "TGCA")
    rc = kmer.translate(complement)[::-1]
    return min(kmer, rc)

def hash_kmer(kmer: str) -> int:
    h = hashlib.md5(kmer.encode('utf-8')).digest()
    return int.from_bytes(h[:4], 'little')

def rolling_kmer_hashes(seq: str, k: int):
    for i, _ in enumerate(seq):
        if i >= k:
            kmer = seq[i-k+1:i+1]
            yield hash_kmer(canonical_kmer(kmer))

def mashmap_map_read(read_seq: str, index: MashMapIndex, min_hits_ratio=0.2):
    hits = index.query(read_seq)
    if not hits:
        return []
    max_offset, max_count = max(hits.items(), key=lambda x: x[1])
    first_w = index.windows[0]
    minimizer_count = sum(1 for _ in get_minimizers(read_seq, k=index.k, w=first_w))
    min_hits = max(1, int(minimizer_count * min_hits_ratio))
    if max_count < min_hits:
        return []
    return [max_offset]


def main():
    seq_rec = next(SeqIO.parse(argv[1], "fasta"))
    genome = str(seq_rec.seq)

    k = choose_k_for_error(0.2)
    index = MashMapIndex(genome, k=k, windows=[15,30,60])

    reads = SeqIO.parse(argv[2], "fasta")
    with open(argv[3], "w") as fout:
        for read in reads:
            read_seq = str(read.seq)
            positions = mashmap_map_read(read_seq, index, min_hits_ratio=0.2)
            for pos in positions:
                fout.write(f"{read.id}\t{pos}\t{pos + len(read_seq)}\n")
                break

if __name__ == "__main__":
    main()
