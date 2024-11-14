import os

import tables as tb
import numpy as np
from scipy.optimize import linear_sum_assignment
import sklearn.metrics

from .utils import calculate_similarity_matrix


def compute_species_center_similarity(
    embeddings: np.array,
    label_ids: np.array,
    results_threshold_similarities_path: str,
    model_name: str,
    percentile_threshold=70,
) -> float:
    """Computes the similarities between each contig and its species center. Concatenates all similarites across species to a list. A specified percentile of all similarities is returned as the threshold. Also saves all similarites results folder.
    N.B: Use the threshold dataset (or other holdout data) for this function.

    Args:
        embeddings (np.array): embeddings from the model
        label_ids (np.array): labels correspondiong to the embeddings
        percentile_threshold (int, optional): The percentile that is used to obtain the threshold from the list of all similarities. Defaults to 70.

    Returns:
        float: Threshold using the specified percentile_threshold
    """

    all_similarities = []
    unique_labels = np.unique(label_ids)
    for label in unique_labels:
        label_ids_filtered = np.where(label_ids == label)[0]
        embeddings_filtered = embeddings[label_ids_filtered]

        label_centroid = np.mean(embeddings_filtered, axis=0)
        similarities_to_centroid = np.dot(embeddings_filtered, label_centroid)

        all_similarities.extend(similarities_to_centroid)

    all_similarities = np.array(all_similarities)
    all_similarities.sort()

    threshold = np.percentile(all_similarities, percentile_threshold)
    print(f"Threshold: {threshold}")

    results_threshold_similarities_file = os.path.join(
        results_threshold_similarities_path,
        model_name + "_" + str(percentile_threshold) + ".npy",
    )

    if not os.path.exists(results_threshold_similarities_file):
        with open(results_threshold_similarities_file, "wb") as f:
            np.save(f, all_similarities)
        print(
            f"Saved threshold similarities to path {results_threshold_similarities_file}"
        )

    return threshold, percentile_threshold


def KMediod(
    embeddings: np.array,
    min_similarity,
    hd5_path: str,
    min_bin_size=10,
    num_steps=3,
    max_iter=1000,
) -> np.array:
    """Performs K-mediod clustering algorithm as described in DNABERT-S (
    https://doi.org/10.48550/arXiv.2402.08777). Code is modified from https://github.com/MAGICS-LAB/DNABERT_S/tree/main.

        Args:
            embeddings (np.array): normalized embeddings with dimensions (n_samples, n_embeddings)
            min_similarity (float, optional): threshold used to decide whether two embeddings are neighbours. Has to be computed for each model.
            hd5_path (str): path to the the similarity matrix in hd5 format.
            min_bin_size (int, optional): minimum binning size; clusters with fewer instances than min_bin_size will be discarded. Defaults to 10.
            num_steps (int, optional): number of steps that the seed is moved. Defaults to 3.
            max_iter (int, optional): number of iterations, where one iteration corresponds to one cluster. Defaults to 1000.

        Returns:
            np.array: predicted predictions for each instance with dimensions (n_samples,)
    """

    n = embeddings.shape[0]

    print(f"Calculating similarity matrix with {n} samples.\n")

    hd5_path = calculate_similarity_matrix(embeddings, min_similarity, hd5_path)

    predictions = np.ones(n) * -1
    predictions = predictions.astype(int)
    print("=========================================\n")
    print(f"Running KMedoid on {n} samples.\n")
    print("=========================================\n")
    with tb.open_file(hd5_path, "r") as f:
        similarities = f.root.similarities

        density_vector = np.sum(similarities, axis=1)

        count = 0
        while np.any(predictions == -1):
            count += 1
            if count > max_iter:
                break
            i = np.argmax(density_vector)
            density_vector[i] = -100  # discards the seed from density vector

            seed = embeddings[i]
            idx_within = np.zeros(len(embeddings), dtype=bool)
            idx_available = predictions == -1

            for _ in range(num_steps):
                similarity = np.dot(embeddings, seed)
                idx_within = similarity >= min_similarity
                idx = np.where(np.logical_and(idx_within, idx_available))[0]
                seed = np.mean(
                    embeddings[idx], axis=0
                )  # seed is the mean of neighbours

            # assign predictions
            predictions[idx] = count
            density_vector -= np.sum(
                similarities[:, idx], axis=1
            )  # updating density vector by the removed indices
            density_vector[idx] = (
                -100
            )  # discards the chosen instances from density vector
            if count % 20 == 0:
                print(f"KMedoid Step {count} completed.")
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


def process_unpredicted_contigs(
    all_predictions: np.array,
    all_labels: np.array,
    embeddings: np.array,
    processing_method: str,
) -> tuple[np.array, np.array, int]:
    """
    Processes unpredicted contigs (contigs assigned -1 by the K-medoid algorithm).
    This function handles unpredicted contigs by either removing them or assigning them
    to the nearest species centroid based on the specified processing method.

    Args:
        all_predictions (np.array): Array of predictions from the K-medoid algorithm,
                                    where -1 indicates unassigned contigs.
        all_labels (np.array): Array of true labels for each contig.
        embeddings (np.array): Array of embeddings for each contig, used to compute
                               distances to centroids if required.
        processing_method (str): Specifies the handling method for unassigned contigs.
                                 Options are:
                                 - "remove": Removes all unassigned contigs.
                                 - "nearest_centroid": Assigns each unassigned contig
                                   to the nearest species centroid.

    Returns:
        tuple[np.array, np.array, int]: A tuple containing:
            - postprocessed_predictions (np.array): Updated predictions array with
              unassigned contigs handled based on the specified method.
            - postprocessed_labels (np.array): Updated labels array, corresponding
              to the postprocessed predictions.
            - n_unpredicted_contigs (int): Count of unassigned contigs before processing.
    """

    n_unpredicted_contigs = len(all_predictions[all_predictions == -1])

    if processing_method == "remove":
        postprocessed_predictions = all_predictions[all_predictions != -1]
        postprosessed_labels = all_labels[all_predictions != -1]

    elif processing_method == "nearest_centroid":

        assert len(all_predictions) == len(all_labels) == len(embeddings)

        # calculate species centroid
        unique_predictions = np.unique(all_predictions[all_predictions != -1])
        all_prediction_centroids = []
        for prediction in unique_predictions:
            predictions_filtered = np.where(all_predictions == prediction)[0]
            embeddings_filtered = embeddings[predictions_filtered]

            prediction_centroid = np.mean(embeddings_filtered, axis=0)
            all_prediction_centroids.append(prediction_centroid)

        all_prediction_centroids = np.array(all_prediction_centroids)

        unassigned_embeddings_indices = np.where(all_predictions == -1)[0]
        unassigned_embeddings = embeddings[all_predictions == -1]

        similarities_to_centroids = np.dot(
            unassigned_embeddings, all_prediction_centroids.T
        )  #  dim (n_unassigned_embeddings, n_centroids)
        nearest_centroid_predictions = unique_predictions[
            np.argmax(similarities_to_centroids, axis=1)
        ]  # dim(n_unclassified_contigs, )

        postprocessed_predictions = all_predictions.copy()
        postprocessed_predictions[unassigned_embeddings_indices] = (
            nearest_centroid_predictions
        )
        postprosessed_labels = all_labels

    return postprocessed_predictions, postprosessed_labels, n_unpredicted_contigs
