"""Main python file to run the pipeline"""

import os
import csv
import traceback
import json
import yaml

import torch
import numpy as np
from sklearn.preprocessing import normalize
import sklearn.metrics as metrics

from src.utils import (
    preprocess_contigs,
    summary_stats,
    label_to_id,
    split_dataset,
)
from src.embeddings import get_embeddings
from src.eval import (
    compute_species_center_similarity,
    KMediod,
    align_labels_via_linear_sum_assignemt,
    compute_eval_metrics,
    process_unpredicted_contigs,
    compute_baseline_metrics,
    calculate_species_distance_matrix
)

import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", message="Increasing alibi size")


def main():
    # Define Paths
    contig_path = "data/metahit/contigs.fna.gz"
    contig_processed_path = "data/species_labelled_contigs.csv"
    threshold_dataset_indices_path = "data/threshold_dataset_indices.npy"
    model_configs = "config/models.yml"

    results_path = "results/main_results"
    results_threshold_similarities_path = "results/threshold_similarities"
    results_heatmap_path = "results/heatmap"

    # Read DNA Sequences
    preprocess_contigs(contig_path, contig_processed_path)
    summary_stats(contig_processed_path)

    # Read Processed DNA Sequences
    with open(contig_processed_path) as csvfile:
        data = list(csv.reader(csvfile, delimiter=","))
    dna_sequences = [i[1] for i in data[1:]]
    label_ids, id2label = label_to_id(data)

    # Read Model Configs
    with open(model_configs, "r") as model_file:
        models_config = yaml.safe_load(model_file)

    compute_baseline_metrics(label_ids, results_path)

    for model_name in list(models_config.keys()):
        print("\n========================================= \n")
        print(f"Using {model_name} to calculate embeddings\n")
        print("========================================= \n\n")
        model_path = models_config[model_name]["model_path"]
        save_path = models_config[model_name]["embedding_path"]
        batch_sizes = models_config[model_name]["batch_sizes"]

        try:
            embeddings = get_embeddings(
                dna_sequences,
                batch_sizes,
                model_name,
                model_path,
                save_path,
            )
            embeddings = normalize(embeddings)
        except Exception:
            print(
                f"|===========| Error in getting embeddings for {model_name}|===========|\n{traceback.format_exc()}"
            )
            continue
        torch.cuda.empty_cache()

        embeddings_evaluate, embeddings_threshold, labels_evaluate, labels_threshold = (
            split_dataset(embeddings, label_ids, threshold_dataset_indices_path)
        )

        threshold, percentile_threshold = compute_species_center_similarity(
            embeddings_threshold,
            labels_threshold,
            results_threshold_similarities_path,
            model_name,
            percentile_threshold=70,
        )

        all_predictions = KMediod(embeddings_evaluate, threshold)

        calculate_species_distance_matrix(embeddings_evaluate, labels_evaluate, results_heatmap_path, model_name)

        print(
            f"Found {len(np.unique(all_predictions))} out of {len(set(label_ids))} "
        )  # Ideal 290

        for pp_method in ["remove", "nearest_centroid"]:
            (
                pp_predictions,
                pp_labels,
                n_unclassified_contigs,
                pp_embeddings,
            ) = process_unpredicted_contigs(
                all_predictions,
                labels_evaluate,
                embeddings_evaluate,
                pp_method,
            )

            print("Aligning labels via linear sum assignment")
            label_mappings = align_labels_via_linear_sum_assignemt(
                pp_labels, pp_predictions
            )

            pp_assigned_labels = [label_mappings[label] for label in pp_predictions]

            print("Calculating silhouette score")
            silhouette_score = metrics.silhouette_score(pp_embeddings, pp_labels)

            compute_eval_metrics(
                pp_labels,
                pp_assigned_labels,
                results_path,
                model_name,
                pp_method,
                percentile_threshold,
                silhouette_score,
                n_unclassified_contigs,
            )

        print("========================================= \n \n")
    return


if __name__ == "__main__":
    main()
