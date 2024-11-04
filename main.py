"""Main python file to run the pipeline"""

import os
import csv
import yaml
import json

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize

from src.utils import (
    preprocess_contigs,
    summary_stats,
    label_to_id,
)
from src.embeddings import get_embeddings
from src.eval import (
    compute_class_center_medium_similarity,
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
    model_configs = "config/models.yml"
    results_path = "results/"

    # Read DNA Sequences
    preprocess_contigs(contig_path, contig_processed_path)
    summary_stats(contig_processed_path)

    # Read Processed DNA Sequences
    with open(contig_processed_path) as csvfile:
        data = list(csv.reader(csvfile, delimiter=","))
    dna_sequences = [i[1] for i in data[1:]]
    label_ids, id2label = label_to_id(data)
    label_ids = label_ids

    # Read Model Configs
    with open(model_configs, "r") as model_file:
        models_config = yaml.safe_load(model_file)

    for model_name in list(models_config.keys()):

        print(f"Using {model_name} to calculate embeddings")

        model_path = models_config[model_name]["model_path"]
        save_path = models_config[model_name]["embedding_path"]

        try:
            embeddings = get_embeddings(
                dna_sequences, model_name, model_path, save_path
            )
            embeddings = normalize(embeddings)
        except Exception as e:
            print(f"Error in getting embeddings for {model_name} with error: {e}")
            continue

        percentile_values, sampled_indices_list = (
            compute_class_center_medium_similarity(embeddings, label_ids)
        )

        threshold = percentile_values[7]
        print(f"threshold: {threshold}")

        predictions = KMediod(embeddings, threshold)
        print(
            f"Found {len(np.unique(predictions))} out of {len(set(label_ids))} "
        )  # Ideal 290

        labels_in_preds = label_ids[predictions != -1]
        predictions = predictions[predictions != -1]

        label_mappings = align_labels_via_linear_sum_assignemt(
            labels_in_preds, predictions
        )
        predictions = [label_mappings[label] for label in predictions]

        results = compute_eval_metrics(labels_in_preds, predictions)

        model_results = {model_name: results}
        model_results_path = os.path.join(results_path, model_name + ".json")
        with open(model_results_path, "w") as results_file:
            json.dump(model_results, results_file)
    return


if __name__ == "__main__":
    main()
