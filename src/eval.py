import numpy as np
from scipy.optimize import linear_sum_assignment
import sklearn.metrics


def compute_class_center_medium_similarity(
    embeddings, labels, sample_rate=0.2, sample_type="class_sample"
):
    """
    Compute similarity metrics by sampling embeddings within each class or among classes.

    Args:
        embeddings (np.ndarray): Embeddings of data points.
        labels (np.ndarray): Class labels for the embeddings.
        sample_rate (float): Proportion of data to sample from each class or number of classes.
        sample_type (str): 'class_sample' for sampling within each class,
                           'classwise_sample' for sampling whole classes.

    Returns:
        tuple: List of percentile values representing similarity distributions,
               List of indices of sampled data points,
               List of labels of sampled classes if sample_type is 'classwise_sample'.
    """
    unique_labels = np.unique(labels)
    all_similarities = []
    sampled_indices_list = []  # To store indices of sampled points

    # Sampled labels only if sample_type is 'classwise_sample'
    sampled_labels = None
    if sample_type == "classwise_sample":
        sampled_labels = np.random.choice(
            unique_labels, int(len(unique_labels) * sample_rate), replace=False
        )

    # Process each class based on the specified sampling strategy
    for label in unique_labels:
        class_indices = np.where(labels == label)[0]
        class_embeddings = embeddings[class_indices]

        # Skip classes with fewer than 10 samples
        if len(class_embeddings) <= 10:
            continue

        if sample_type == "class_sample":
            # Sample a percentage of data points within the class
            n_sample = max(1, int(len(class_embeddings) * sample_rate))
            sampled_indices = np.random.choice(
                len(class_embeddings), n_sample, replace=False
            )
            sampled_embeddings = class_embeddings[sampled_indices]
            sampled_original_indices = class_indices[sampled_indices]

        elif sample_type == "classwise_sample":
            # Check if the current class is in the sampled labels
            if label not in sampled_labels:
                continue
            sampled_embeddings = class_embeddings
            sampled_original_indices = class_indices  # Use all indices in this class

        # Calculate the mean and similarities for the sampled embeddings
        mean_embedding = np.mean(sampled_embeddings, axis=0)
        similarities = np.dot(sampled_embeddings, mean_embedding)

        # Collect all similarities and sampled indices for percentile calculation
        all_similarities.extend(similarities)
        sampled_indices_list.extend(
            sampled_original_indices
        )  # Add original indices to the list

    # Compute and return percentile values for the aggregated similarities
    all_similarities = np.array(all_similarities)
    all_similarities.sort()
    percentile_values = [
        np.percentile(all_similarities, p) for p in [10, 20, 30, 40, 50, 60, 70, 80, 90]
    ]
    return percentile_values, sampled_indices_list


def KMediod(
    embeddings: np.array,
    min_similarity=0.8,
    min_bin_size=10,
    num_steps=3,
    max_iter=1000,
) -> np.array:
    """Performs K-mediod clustering algorithm as described in DNABERT-S (
    https://doi.org/10.48550/arXiv.2402.08777). Code is modified from https://github.com/MAGICS-LAB/DNABERT_S/tree/main.

        Args:
            embeddings (np.array): normalized embeddings with dimensions (n_samples, n_embeddings)
            min_similarity (float, optional): threshold used to decide whether two embeddings are neighbours. Has to be computed for each model. Defaults to 0.8.
            min_bin_size (int, optional): minimum binning size; clusters with fewer instances than min_bin_size will be discarded. Defaults to 10.
            num_steps (int, optional): number of steps that the seed is moved. Defaults to 3.
            max_iter (int, optional): number of iterations, where one iteration corresponds to one cluster. Defaults to 1000.

        Returns:
            np.array: predicted predictions for each instance with dimensions (n_samples,)
    """
    embeddings = embeddings.astype(np.float16)
    similarities = np.dot(embeddings, embeddings.T)  # EE^T

    similarities[similarities < min_similarity] = 0
    density_vector = np.sum(similarities, axis=1)

    predictions = np.ones(len(embeddings)) * -1
    predictions = predictions.astype(int)
    count = 0
    print(f"Running KMedoid on {embeddings.shape[0]} samples.")
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
            seed = np.mean(embeddings[idx], axis=0)  # seed is the mean of neighbours

        # assign predictions
        predictions[idx] = count
        density_vector -= np.sum(
            similarities[:, idx], axis=1
        )  # updating density vector by the removed indices
        density_vector[idx] = -100  # discards the chosen instances from density vector
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
