import numpy as np

def compute_class_center_medium_similarity(embeddings, labels, sample_rate=0.2, sample_type='class_sample'):
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
    if sample_type == 'classwise_sample':
        sampled_labels = np.random.choice(unique_labels, int(len(unique_labels) * sample_rate), replace=False)

    # Process each class based on the specified sampling strategy
    for label in unique_labels:
        class_indices = np.where(labels == label)[0]
        class_embeddings = embeddings[class_indices]

        # Skip classes with fewer than 10 samples
        if len(class_embeddings) <= 10:
            continue

        if sample_type == 'class_sample':
            # Sample a percentage of data points within the class
            n_sample = max(1, int(len(class_embeddings) * sample_rate))
            sampled_indices = np.random.choice(len(class_embeddings), n_sample, replace=False)
            sampled_embeddings = class_embeddings[sampled_indices]
            sampled_original_indices = class_indices[sampled_indices]

        elif sample_type == 'classwise_sample':
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
        sampled_indices_list.extend(sampled_original_indices)  # Add original indices to the list

    # Compute and return percentile values for the aggregated similarities
    all_similarities = np.array(all_similarities)
    all_similarities.sort()
    percentile_values = [np.percentile(all_similarities, p) for p in [10, 20, 30, 40, 50, 60, 70, 80, 90]]

    print(percentile_values)
    return percentile_values, sampled_indices_list


    def KMediod(
    features: np.array, min_similarity=0.8, min_bin_size=100, num_steps=3, max_iter=1000
) -> np.array:
    """
    Performs the K-medoid clustering algorithm as described in DNABERT-S.
    Original code modified from: https://github.com/MAGICS-LAB/DNABERT_S/tree/main.

    Args:
        features (np.array): Normalized embeddings with shape (n_samples, n_features).
        min_similarity (float, optional): Threshold to decide if two embeddings are neighbors.
        min_bin_size (int, optional): Minimum cluster size; clusters smaller than this will be discarded.
        num_steps (int, optional): Number of steps to move the seed.
        max_iter (int, optional): Maximum number of iterations; one iteration corresponds to one cluster.

    Returns:
        np.array: Predicted cluster labels for each instance with shape (n_samples,).
    """

    features = features.astype(np.float32)
    n_samples = features.shape[0]
    block_size = 1000  # Adjust according to your available memory

    # Initialize the density vector
    density_vector = np.zeros(n_samples, dtype=np.float32)

    # Compute the density vector using block processing
    for i in range(0, n_samples, block_size):
        end_i = min(i + block_size, n_samples)
        features_block_i = features[i:end_i]

        for j in range(0, n_samples, block_size):
            end_j = min(j + block_size, n_samples)
            features_block_j = features[j:end_j]

            # Compute partial similarity matrix
            similarities_block = np.dot(features_block_i, features_block_j.T)

            # Zero out similarities below the threshold
            similarities_block[similarities_block < min_similarity] = 0

            # Update the density vector
            density_vector[i:end_i] += np.sum(similarities_block, axis=1)

    predictions = np.full(n_samples, -1, dtype=int)
    count = 0

    while np.any(predictions == -1):
        count += 1
        if count > max_iter:
            break

        # Get the index with the maximum density
        i = np.argmax(density_vector)
        density_vector[i] = -100  # Exclude the seed from the density vector

        seed = features[i]
        idx_available = predictions == -1

        for _ in range(num_steps):
            similarity = np.zeros(n_samples, dtype=np.float32)

            # Compute similarities between the seed and other points using block processing
            for j in range(0, n_samples, block_size):
                end_j = min(j + block_size, n_samples)
                features_block = features[j:end_j]

                similarities_block = np.dot(features_block, seed)
                similarity[j:end_j] = similarities_block

            idx_within = similarity >= min_similarity
            idx = np.where(np.logical_and(idx_within, idx_available))[0]

            if len(idx) == 0:
                break

            # Update the seed to be the mean of neighboring points
            seed = np.mean(features[idx], axis=0)

        # Update predictions
        predictions[idx] = count

        # Update the density vector by removing the influence of selected indices
        for i_dv in range(0, n_samples, block_size):
            end_i_dv = min(i_dv + block_size, n_samples)
            features_block_i = features[i_dv:end_i_dv]

            # Compute similarities with the selected indices
            for j in idx:
                sim_block = np.dot(features_block_i, features[j])

                # Zero out similarities below the threshold
                sim_block[sim_block < min_similarity] = 0

                density_vector[i_dv:end_i_dv] -= sim_block

        density_vector[idx] = -100  # Exclude selected indices from the density vector
        print(f"Cluster {count} completed.")

    # Remove clusters that are too small
    unique, counts = np.unique(predictions, return_counts=True)
    for i, c in zip(unique, counts):
        if c < min_bin_size:
            predictions[predictions == i] = -1

    return predictions
