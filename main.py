"""Main python file to run the pipeline"""

import os
import csv
import traceback
import json
import yaml

import torch
import numpy as np
from sklearn.preprocessing import normalize

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

    binning_results_path = "results/binning"
    results_threshold_similarities_path = "results/threshold_similarities"

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

    for model_name in list(models_config.keys()):
        print("\n========================================= \n")
        print(f"Using {model_name} to calculate embeddings\n")
        print("========================================= \n\n")
        model_path = models_config[model_name]["model_path"]
        save_path = models_config[model_name]["embedding_path"]
        hd5_path = models_config[model_name]["similarities_path"]
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

        all_predictions = KMediod(embeddings_evaluate, threshold, hd5_path)

        print(
            f"Found {len(np.unique(all_predictions))} out of {len(set(label_ids))} "
        )  # Ideal 290

        labels_in_preds = labels_evaluate[all_predictions != -1]
        valid_predictions = all_predictions[all_predictions != -1]

        label_mappings = align_labels_via_linear_sum_assignemt(
            labels_in_preds, valid_predictions
        )
        valid_predictions = [label_mappings[label] for label in valid_predictions]

        results = compute_eval_metrics(labels_in_preds, valid_predictions)

        model_results = {
            model_name: {
                "percentile threshold": percentile_threshold,
                "results": results,
            }
        }
        model_results_path = os.path.join(binning_results_path, model_name + ".json")
        with open(model_results_path, "w") as results_file:
            json.dump(model_results, results_file)
        print("========================================= \n \n")
    return


if __name__ == "__main__":
    main()
