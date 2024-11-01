import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from utils import (
    preprocess_contigs,
    KMediod,
    align_labels_via_linear_sum_assignemt,
    compute_eval_metrics,
)

from threshold_eisuke import compute_class_center_medium_similarity


def main():
    # file_path = "metahit_data/contigs.fna.gz"
    # preprocess_contigs(file_path, "contigs.csv")

    file_path = "contigs.csv"
    contigs = pd.read_csv(file_path)
    contigs = contigs.iloc[:50000, :]
    print(contigs.shape)

    labels = contigs.iloc[:, 2]
    print(labels[0])

    # Convert label to numeric ID
    label2id = {l: i for i, l in enumerate(set(labels))}
    labels_bin = np.array([label2id[l] for l in labels])
    print(labels_bin[0])

    embeddings = np.load("hyenadna.npy")[:50000]
    
     embedding_norm = normalize(embeddings)

    percentile_values, sampled_indices_list = compute_class_center_medium_similarity(
        embeddings, labels
    )

    threshold = percentile_values[-3]
    print(threshold)

   

    binning_results = KMediod(
        embedding_norm, min_similarity=threshold, min_bin_size=10, max_iter=1000
    )
    print(len(np.unique(binning_results)))

    true_labels_bin = labels_bin[binning_results != -1]
    predicted_labels = binning_results[binning_results != -1]

    print(true_labels_bin)
    print(predicted_labels)
    # Align labels
    # alignment_bin = align_labels_via_linear_sum_assignemt(
    #    true_labels_bin, predicted_labels
    # )
    # predicted_labels_bin = [alignment_bin[label] for label in predicted_labels]

    # result = compute_eval_metrics(true_labels_bin, predicted_labels_bin)
    # print(result)


if __name__ == "__main__":
    main()
