import os
import json

import numpy as np
from scipy.optimize import linear_sum_assignment
import sklearn.metrics

import tqdm


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


def calculate_species_distance_matrix(embeddings, ids, path: str, model_name: str) -> None:
    """
    Calculate the distance matrix for groups of embeddings based on labels.

    Parameters:
    embeddings (np.ndarray): N*d array of embeddings.
    ids (np.ndarray): N*1 array of sorted ids.
    path (str): Directory path where the result JSON file will be saved.
    model_name (str): Name of the model, used for naming the JSON file and as a key in the JSON content.
    

    Returns:
    None
    """
    # Generate the file path for saving the results
    model_results_path = os.path.join(path, model_name + "_dist_mtx")

    # Remove entries with specific ids (287 and 288)
    mask = (ids != 287) & (ids != 288)
    embeddings = embeddings[mask]
    ids = ids[mask]

    # Get unique ids
    unique_ids = np.unique(ids)
    num_ids = len(unique_ids)

    # Initialize distance matrix
    distance_matrix = np.zeros((num_ids, num_ids))

    # Compute Hausdorff distance between species
    for i, id1 in enumerate(unique_ids):
        for j, id2 in enumerate(unique_ids):
            if i <= j:  # Distance matrix is symmetric
                species1 = embeddings[ids==id1]
                species2 = embeddings[ids==id2]

                # Compute the distant matrix for each speicies
                dist_mtrx = euclidean_distances(species1, species2)
                dist_1 = np.percentile(dist_mtrx.min(axis=0), 95)
                dist_2 = np.percentile(dist_mtrx.min(axis=1), 95)

                # Calculate Hausdorff distance
                distance = max(dist_1, dist_2)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance  # Symmetry

    np.save(model_results_path, distance_matrix)



def KMediod(
    embeddings: np.array,
    min_similarity=0.8,
    min_bin_size=100,
    num_steps=3,
    max_iter=1000,
) -> np.array:
    """
    Performs the K-medoid clustering algorithm as described in DNABERT-S.
    Original code modified from: https://github.com/MAGICS-LAB/DNABERT_S/tree/main.

    Args:
        embeddings (np.array): Normalized embeddings with shape (n_samples, d).
        min_similarity (float, optional): Threshold to decide if two embeddings are neighbors.
        min_bin_size (int, optional): Minimum cluster size; clusters smaller than this will be discarded.
        num_steps (int, optional): Number of steps to move the seed.
        max_iter (int, optional): Maximum number of iterations; one iteration corresponds to one cluster.

    Returns:
        np.array: Predicted cluster labels for each instance with shape (n_samples,).
    """

    embeddings = embeddings.astype(np.float32)
    n_samples = embeddings.shape[0]
    block_size = 1000  # Adjust according to your available memory

    # Initialize the density vector
    density_vector = np.zeros(n_samples, dtype=np.float32)

    # Compute the density vector using block processing
    for i in tqdm.tqdm(
        range(0, n_samples, block_size), desc="Computing Density Vector"
    ):
        end_i = min(i + block_size, n_samples)
        embeddings_block_i = embeddings[i:end_i]

        for j in range(0, n_samples, block_size):
            end_j = min(j + block_size, n_samples)
            embeddings_block_j = embeddings[j:end_j]

            # Compute partial similarity matrix
            similarities_block = np.dot(embeddings_block_i, embeddings_block_j.T)

            # Zero out similarities below the threshold
            similarities_block[similarities_block < min_similarity] = 0

            # Update the density vector
            density_vector[i:end_i] += np.sum(similarities_block, axis=1)

    predictions = np.full(n_samples, -1, dtype=int)
    count = 0
    print("=========================================\n")
    print(f"Running KMedoid on {n_samples} samples.\n")
    print("=========================================\n")
    progress_bar = tqdm.tqdm(total=max_iter, desc="KMedoid Progress")
    while np.any(predictions == -1):
        count += 1
        if count > max_iter:
            break

        # Get the index with the maximum density
        i = np.argmax(density_vector)
        density_vector[i] = -100  # Exclude the seed from the density vector

        seed = embeddings[i]
        idx_available = predictions == -1

        for _ in range(num_steps):
            similarity = np.zeros(n_samples, dtype=np.float32)

            # Compute similarities between the seed and other points using block processing
            for j in range(0, n_samples, block_size):
                end_j = min(j + block_size, n_samples)
                embeddings_block = embeddings[j:end_j]

                similarities_block = np.dot(embeddings_block, seed)
                similarity[j:end_j] = similarities_block

            idx_within = similarity >= min_similarity
            idx = np.where(np.logical_and(idx_within, idx_available))[0]

            if len(idx) == 0:
                break

            # Update the seed to be the mean of neighboring points
            seed = np.mean(embeddings[idx], axis=0)

        # Update predictions
        predictions[idx] = count

        # Update the density vector by removing the influence of selected indices
        for i_dv in range(0, n_samples, block_size):
            end_i_dv = min(i_dv + block_size, n_samples)
            embeddings_block_i = embeddings[i_dv:end_i_dv]

            # Compute similarities with the selected indices
            for j in idx:
                sim_block = np.dot(embeddings_block_i, embeddings[j])

                # Zero out similarities below the threshold
                sim_block[sim_block < min_similarity] = 0

                density_vector[i_dv:end_i_dv] -= sim_block

        density_vector[idx] = -100  # Exclude selected indices from the density vector
        progress_bar.update(count)
        if count % 20 == 0:
            print(f"KMedoid Step {count} completed.")

    # Remove clusters that are too small
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


def compute_eval_metrics(
    true_labels: np.array,
    predicted_labels: np.array,
    results_path: str,
    model_name: str,
    postprocessing_method: str,
    percentile_threshold: float,
    silhouette_score: float,
    n_unclassified_contigs: int,
) -> None:
    """
    Calculates recall and F1 score, and provides results at thresholds.

    Args:
        true_labels_bin (np.array): True labels.
        predicted_labels_bin (np.array): Predicted labels that are aligned to the true labels (e.g. using align_labels_via_linear_sum_assignemt).
        path (str): Directory path where the result JSON file will be saved.
        model_name (str): Name of the model, used for naming the JSON file and as a key in the JSON content.

    Returns:
        None
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

    model_results = {
        model_name: {
            "percentile_threshold": percentile_threshold,
            "thresholds": thresholds,
            "f1_results": f1_results,
            "recall_results": recall_results,
            "silhouette_score": float(silhouette_score),
            "n_unclassified_contigs": n_unclassified_contigs,
        }
    }

    model_results_path = os.path.join(
        results_path, model_name + "_" + postprocessing_method + ".json"
    )

    with open(model_results_path, "w") as results_file:
        json.dump(model_results, results_file)
    return


def compute_baseline_metrics(true_labels: list, results_path: str) -> None:
    """Compute f1 and recall for shuffled predicted labels.

    Args:
        true_labels_bin (list): True labels.
        path (str): Directory path where the result JSON file will be saved.

    Returns:
        None
    """

    true_labels = np.array(true_labels)
    shuffled_predicted_labels = np.random.randint(
        1, len(np.unique(true_labels)) + 1, size=len(true_labels)
    )
    assert len(shuffled_predicted_labels) == len(true_labels)

    # Calculate recall for each class
    recall_bin = sklearn.metrics.recall_score(
        true_labels, shuffled_predicted_labels, average="macro", zero_division=0
    )

    # Calculate F1 scores for each class
    f1_bin = sklearn.metrics.f1_score(
        true_labels, shuffled_predicted_labels, average="macro", zero_division=0
    )

    baseline_results = {"f1_baseline": f1_bin, "recall_baseline": recall_bin}

    baseline_results_path = os.path.join(results_path, "baseline_results" + ".json")

    with open(baseline_results_path, "w") as results_file:
        json.dump(baseline_results, results_file)
    return


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
    if processing_method == "remove":
        n_unpredicted_contigs = len(all_predictions[all_predictions == -1])
        postprocessed_predictions = all_predictions[all_predictions != -1]
        postprosessed_labels = all_labels[all_predictions != -1]
        postprocessed_embeddings = embeddings[all_predictions != -1]

        return (
            postprocessed_predictions,
            postprosessed_labels,
            n_unpredicted_contigs,
            postprocessed_embeddings,
        )

    elif processing_method == "nearest_centroid":

        assert len(all_predictions) == len(all_labels) == len(embeddings)
        n_unpredicted_contigs = 0
        # calculate species centroid
        unique_predictions = np.unique(all_predictions[all_predictions != -1])
        all_prediction_centroids = []
        for prediction in tqdm.tqdm(
            unique_predictions,
            desc="Calculating Nearest Centroids to Unassigned Contigs",
        ):
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

        # postprocessed_predictions = all_predictions.copy()
        all_predictions[unassigned_embeddings_indices] = nearest_centroid_predictions

        return all_predictions, all_labels, n_unpredicted_contigs, embeddings
