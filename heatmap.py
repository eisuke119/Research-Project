import os
import csv
import traceback
import json
import yaml

import torch
import numpy as np
import pandas as pd
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
    align_labels_via_linear_sum_assignemt,
    compute_eval_metrics,
    calculate_species_distance_matrix,
    plot_hierarchical_clustering_with_labels,
    create_tsne_plot
)

import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", message="Increasing alibi size")


def main():
    # Define Paths
    contig_path = "data/metahit/contigs.fna.gz"
    taxonomy_path = "data/metahit/taxonomy.tsv"
    contig_processed_path = "data/species_labelled_contigs.csv"
    threshold_dataset_indices_path = "data/threshold_dataset_indices.npy"
    model_configs = "config/models.yml"

    results_path = "results/heatmap"

    # Read DNA Sequences
    preprocess_contigs(contig_path, contig_processed_path)

    # Read Processed DNA Sequences
    with open(contig_processed_path) as csvfile:
        data = list(csv.reader(csvfile, delimiter=","))
    dna_sequences = [i[1] for i in data[1:]]
    label_ids, id2label = label_to_id(data)

    # Read Taxonomy
    taxon = pd.read_csv(taxonomy_path, delimiter='\t', header=None)
    species2genus = dict(zip(taxon[0], taxon[2]))
    id2genus = {i: label for i, label in enumerate(taxon[2])}
    genus2id = {label: i for i, label in enumerate(taxon[2])}

    # Read Model Configs
    with open(model_configs, "r") as model_file:
        models_config = yaml.safe_load(model_file)

    for model_name in list(models_config.keys()):
        output_path = os.path.join(results_path, model_name + "_heatmap.png")
        if os.path.exists(output_path):
            continue
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

        calculate_species_distance_matrix(embeddings_evaluate, labels_evaluate, results_path, model_name)
        

        print("========================================= \n \n")
    return


if __name__ == "__main__":
    main()