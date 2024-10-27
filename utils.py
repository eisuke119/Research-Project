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
