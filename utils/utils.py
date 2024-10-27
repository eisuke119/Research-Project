import re
import gzip
import collections
import numpy as np
from Bio import SeqIO
from scipy.optimize import linear_sum_assignment
import sklearn.metrics


def load_contigs(file_path: str, min_length=2500, max_length=100000) -> np.array:
    """Load contig data and their IDs.
    Only keeps sequences with: min_length < seq_length < max_length
    Example ID that is parsed: gi|224815735|ref|NZ_ACGB01000001.1|_[Acidaminococcus_D21_uid55871]_1-5871

    Args:
        file_path (str): file path for contigs in format .fna.gz
        min_length (int, optional): minimum length for the DNA strings. Defaults to 2500.
        max_length (int, optional): maximum length for the DNA strings. Defaults to 100000.

    Returns:
            np.array: returns 1 array with numeric labels and 1 array with dna-sequences

    Raises:
        ValueError: When metadata from the fasta-id can not be parsed correctly.

    """

    print(f"***** Start loading data from file path: {file_path} *****")

    sequences = {}
    # Regular expression to capture metadata from the FASTA header, e.g. gi|224815735|ref|NZ_ACGB01000001.1|_[Acidaminococcus_D21_uid55871]_1-5871
    pattern = re.compile(r"gi\|[\d]+\|ref\|(.*?)\.(\d)\|_\[(.*?)\]_(\d+)-(\d+)")

    with gzip.open(file_path, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            match = pattern.search(record.id)
            if match:
                scaffold_id = int(match.group(2))
                species = match.group(3)
                start = int(match.group(4))
                end = int(match.group(5))

                sequences[record.id] = {
                    "scaffold_id": scaffold_id,
                    "species": species,
                    "start": start,
                    "end": end,
                    "length": len(record.seq),
                    "sequence": str(record.seq),
                }

            else:
                raise ValueError(f"Could not extract metadata from ID: {record.id}")

    labels = [seq["species"] for seq in sequences.values()]
    dna_sequences = [seq["sequence"] for seq in sequences.values()]

    # Filtering based on length
    initial_count = len(sequences)
    sequences = {
        id_: seq
        for id_, seq in sequences.items()
        if min_length <= seq["length"] <= max_length
    }
    removed_count = initial_count - len(sequences)
    print(
        f"Removed {removed_count} sequences that were shorter than {min_length} or longer than {max_length}."
    )

    # Check nubmer of contigs in each species
    label_counts = list(collections.Counter(labels).values())
    print(
        f"Number of contigs per species; Min: {min(label_counts)}, Max: {max(label_counts)}, Median: {np.median(label_counts)}"
    )

    # Convert label to numeric ID
    label2id = {l: i for i, l in enumerate(set(labels))}
    labels_id = np.array([label2id[l] for l in labels])
    print(f"Got {len(sequences)} sequences, {len(label2id)} clusters")

    return np.array(labels_id), np.array(dna_sequences)


def KMediod(
    features: np.array, min_similarity=0.8, min_bin_size=100, num_steps=3, max_iter=1000
) -> np.array:
    """Performs K-mediod clustering algorithm as described in DNABERT-S (
    https://doi.org/10.48550/arXiv.2402.08777). Code is modified from https://github.com/MAGICS-LAB/DNABERT_S/tree/main.

        Args:
            features (np.array): normalized embeddings with dimensions (n_samples, n_features)
            min_similarity (float, optional): threshold used to decide whether two embeddings are neighbours. Has to be computed for each model. Defaults to 0.8.
            min_bin_size (int, optional): minimum binning size; clusters with fewer instances than min_bin_size will be discarded. Defaults to 100.
            num_steps (int, optional): number of steps that the seed is moved. Defaults to 3.
            max_iter (int, optional): number of iterations, where one iteration corresponds to one cluster. Defaults to 1000.

        Returns:
            np.array: predicted predictions for each instance with dimensions (n_samples,)
    """

    features = features.astype(np.float32)
    similarities = np.dot(features, features.T)  # EE^T

    similarities[similarities < min_similarity] = 0
    density_vector = np.sum(similarities, axis=1)

    predictions = np.ones(len(features)) * -1
    predictions = predictions.astype(int)
    count = 0

    while np.any(predictions == -1):
        count += 1
        if count > max_iter:
            break
        i = np.argmax(density_vector)
        density_vector[i] = -100  # discards the seed from density vector

        seed = features[i]
        idx_within = np.zeros(len(features), dtype=bool)
        idx_available = predictions == -1

        for _ in range(num_steps):
            similarity = np.dot(features, seed)
            idx_within = similarity >= min_similarity
            idx = np.where(np.logical_and(idx_within, idx_available))[0]
            seed = np.mean(features[idx], axis=0)  # seed is the mean of neighbours

        # assign predictions
        predictions[idx] = count
        density_vector -= np.sum(
            similarities[:, idx], axis=1
        )  # updating density vector by the removed indices
        density_vector[idx] = -100  # discards the chosen instances from density vector

    # remove bins that are too small
    unique, counts = np.unique(predictions, return_counts=True)
    for i, c in zip(unique, counts):
        if c < min_bin_size:
            predictions[predictions == i] = -1

    return predictions


def align_labels_via_linear_sum_assignemt(
    true_labels: np.array, predicted_labels: np.array
) -> dict:
    """
    Aligns the predicted labels with the true labels using the Linear Sum Assignment Algorithm, i.e. Hungarian algorithm.
    Optimses the maximum weight bipartite graph.
    Code is modified from https://github.com/MAGICS-LAB/DNABERT_S/tree/main.


    Args:
    true_labels (np.array): The true labels of the data.
    predicted_labels (np.array): The labels predicted by k-mediod clustering algorithm.

    Returns:
    dict: A dictionary mapping the predicted labels to the aligned true labels.
    """
    # Create a confusion matrix
    max_label = max(max(true_labels), max(predicted_labels)) + 1
    confusion_matrix = np.zeros((max_label, max_label), dtype=int)

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        confusion_matrix[true_label, predicted_label] += 1

    # Apply the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(
        confusion_matrix, maximize=True
    )  # Optimses the maximum weight bipartite graph

    # Create a mapping from predicted labels to true labels
    label_mapping = {
        predicted_label: true_label
        for true_label, predicted_label in zip(row_ind, col_ind)
    }

    return label_mapping


def compute_eval_metrics(true_labels: np.array, predicted_labels: np.array) -> dict:
    """
    Calculates recall and F1 score, and provides results at thresholds.

    Args:
        true_labels_bin (np.array): True labels.
        predicted_labels_bin (np.array): Predicted labels that are aligned to the true labels (e.g. using align_labels_via_linear_sum_assignemt).

    Returns:
        dict: Contains recall and F1 counts above thresholds.
    """

    # Calculate recall for each class
    recall_bin = sklearn.metrics.recall_score(
        true_labels, predicted_labels, average=None, zero_division=0
    )
    recall_bin.sort()

    # Calculate F1 scores for each class
    f1_bin = sklearn.metrics.f1_score(
        true_labels, predicted_labels, average=None, zero_division=0
    )
    f1_bin.sort()

    # Define thresholds for evaluation
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    recall_results = []
    f1_results = []

    # Evaluate recall and F1 scores above each threshold
    for threshold in thresholds:
        recall_results.append(len(np.where(recall_bin > threshold)[0]))
        f1_results.append(len(np.where(f1_bin > threshold)[0]))

    return {
        "thresholds": thresholds,
        "f1_results": f1_results,
        "recall_results": recall_results,
    }
