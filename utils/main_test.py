import argparse
import os
import sys

from utils import (
    load_contigs,
    KMediod,
    align_labels_via_linear_sum_assignemt,
    compute_eval_metrics,
)


def main(args):

    labels, dna_sequences = load_contigs(args.data_dir)

    print(len(labels))
    print(len(dna_sequences))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate clustering")
    # parser.add_argument('--test_model_dir', type=str, default="/root/trained_model", help='Directory to save trained models to test')
    # parser.add_argument('--model_list', type=str, default="test", help='List of models to evaluate, separated by comma. Currently support [tnf, tnf-k, dnabert2, hyenadna, nt, test]')
    parser.add_argument(
        "--data_dir", type=str, default="/root/data", help="Data directory"
    )
    args = parser.parse_args()
    main(args)
