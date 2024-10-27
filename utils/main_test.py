import argparse
import os
import sys

from utils import (
    load_contigs,
    KMediod,
    align_labels_via_linear_sum_assignemt,
    compute_eval_metrics,
)


def main():
    file_path = "metahit_data/contigs.fna.gz"

    labels, dna_sequences = load_contigs(file_path)

    print(len(labels))
    print(len(dna_sequences))


if __name__ == "__main__":
    main()
