import os

import tables as tb
import numpy as np
from scipy.optimize import linear_sum_assignment
import sklearn.metrics

from .utils import calculate_similarity_matrix

from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import seaborn as sns
from scipy.cluster.hierarchy import fcluster
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D




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
    density_vector = np.zeros(n)
    print("=========================================\n")
    print(f"Running KMedoid on {n} samples.\n")
    print("=========================================\n")
    with tb.open_file(hd5_path, "r") as f:
        similarities = f.root.similarities

        density_vector[:] = np.sum(similarities[:, :], axis=1)

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


def compute_eval_metrics(true_labels: np.array, predicted_labels: np.array, path: str, model_name: str) -> None:
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
                "thresholds": thresholds,
                "f1_results": f1_results,
                "recall_results": recall_results
            }
        }
    
    model_results_path = os.path.join(path, model_name + ".json")
        
    with open(model_results_path, "w") as results_file:
        json.dump(model_results, results_file)

def compute_silhouette_score(embeddings: np.ndarray, predicted_labels: np.ndarray, path: str, model_name: str) -> None:
    """
    Calculates the silhouette score based on embeddings and predicted labels, and saves the result as a JSON file.

    Args:
        embeddings (np.ndarray): Embedding vectors representing the data points.
        predicted_labels (np.ndarray): Predicted cluster labels for each data point.
        path (str): Directory path where the result JSON file will be saved.
        model_name (str): Name of the model, used for naming the JSON file and as a key in the JSON content.

    Returns:
        None
    """

    # Calculate the silhouette score
    score = sklearn.metrics.silhouette_score(embeddings, predicted_labels)

    # Organize the result into a dictionary
    model_ss = {
        model_name: {
            "silhouette_score": score,
        }
    }

    # Generate the file path for saving the results
    model_ss_path = os.path.join(path, model_name + ".json")

    # Save the results as a JSON file
    with open(model_ss_path, "w") as results_file:
        json.dump(model_ss, results_file)


def calculate_species_distance_matrix(embeddings, ids) -> tuple[np.array, np.array]:
    """
    Calculate the similarity matrix for groups of embeddings based on labels.

    Parameters:
    embeddings (np.ndarray): N*d array of embeddings.
    ids (np.ndarray): N*1 array of sorted ids.

    Returns:
    np.ndarray: Distance matrix of size (number of unique ids) * (number of unique ids).
    np.ndarray: Unique ids in order (number of unique ids) * 1.
    """
    # Get unique ids
    unique_ids, idx = np.unique(ids, return_index=True)
    num_ids = len(unique_ids)

    unique_id_ordered = ids[np.sort(idx)]

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
                dist_1 = dist_mtrx.min(axis=0).max()
                dist_2 = dist_mtrx.min(axis=1).max()

                # Calculate Hausdorff distance
                distance = max(dist_1, dist_2)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance  # Symmetry

    return (distance_matrix, unique_id_ordered)


def plot_hierarchical_clustering_with_labels(distance_matrix, path: str, model_name: str, color_threshold=0.7):
    """
    Generate a heatmap with hierarchical clustering based on an N x N distance matrix,
    and return the cluster labels as an N x 1 array.

    Parameters:
    distance_matrix (numpy.ndarray): A symmetric N x N matrix representing pairwise distances.
    path (str): Directory path where the result JSON file will be saved.
    model_name (str): Name of the model, used for naming the JSON file and as a key in the JSON content.
    color_threshold (float): Threshold for coloring clusters in the dendrogram.

    Returns:
    numpy.ndarray: An N x 1 array of cluster labels.
    """
    # Generate the file path for saving the results
    model_results_path = os.path.join(path, model_name + "_heatmap.png")

    # Create a hierarchical linkage matrix
    linkage = sch.linkage(distance_matrix, method='ward')

    # Create cluster labels using fcluster
    cluster_labels = fcluster(linkage, t=color_threshold, criterion='distance')

    # Create a figure for the plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 8), gridspec_kw={'width_ratios': [1, 5]})

    # Plot the dendrogram on the left with custom color scheme
    dendrogram = sch.dendrogram(
        linkage, 
        orientation='left', 
        ax=ax[0], 
        no_labels=True, 
        color_threshold=color_threshold
    )
    ax[0].axis('off')

    # Custom color mapping for the dendrogram (coolwarm)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(dendrogram['color_list'])))
    for color, line in zip(dendrogram['color_list'], ax[0].collections):
        line.set_color(colors[np.random.randint(0, len(colors))])

    # Sort the matrix rows/columns according to the clustering result
    sorted_idx = sch.leaves_list(linkage)
    sorted_matrix = distance_matrix[sorted_idx][:, sorted_idx]

    # Create a heatmap of the sorted matrix
    sns.heatmap(sorted_matrix, ax=ax[1], cmap='coolwarm', cbar=True, xticklabels=False, yticklabels=False)
    ax[1].set_title("Heatmap with Hierarchical Clustering")

    # Adjust layout
    plt.tight_layout()

    # Save the image to the specified file
    plt.savefig(model_results_path, dpi=300)
    plt.close()

    # Return the cluster labels
    return cluster_labels


def create_tsne_plot(distance_matrix, path: str, model_name: str) -> None:
    """
    Computes a 3D t-SNE embedding from a given N x N distance matrix and saves the plot.

    Parameters:
    distance_matrix (numpy.ndarray): A square N x N distance matrix.
    path (str): Directory path where the result JSON file will be saved.
    model_name (str): Name of the model, used for naming the JSON file and as a key in the JSON content.
    
    Returns:
    None
    """
    # Generate the file path for saving the results
    model_results_path = os.path.join(path, model_name + "_tsne.png")

    # Initialize t-SNE with 3 components and precomputed metric
    tsne = TSNE(n_components=3, metric='precomputed', random_state=42, init='random')

    # Compute the t-SNE embedding
    tsne_results = tsne.fit_transform(distance_matrix)

    # Create a 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot the t-SNE results
    scatter = ax.scatter(
        tsne_results[:, 0],  # X-axis values
        tsne_results[:, 1],  # Y-axis values
        tsne_results[:, 2],  # Z-axis values
        c='blue',             # Color of the points
        s=50,                 # Size of the points
        alpha=0.6             # Transparency of the points
    )

    # Set plot title and axis labels with increased labelpad
    ax.set_title('3D t-SNE Visualization', fontsize=18, pad=20)
    ax.set_xlabel('TSNE Dimension 1', fontsize=12, labelpad=10)
    ax.set_ylabel('TSNE Dimension 2', fontsize=12, labelpad=10)
    ax.set_zlabel('TSNE Dimension 3', fontsize=12, labelpad=10)

    # Optionally, adjust the viewing angle for better visualization
    ax.view_init(elev=30, azim=45)

    # Adjust the layout to make room for the z-axis label
    plt.tight_layout(pad=2.0)  # Added padding to ensure labels are within the figure

    # Save the plot to the specified path with additional padding
    plt.savefig(model_results_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close(fig)  # Close the figure to free memory