import gzip
import os
import tables as tb

import tqdm
import numpy as np
from Bio import SeqIO
import pandas as pd

np.random.seed(42)


def preprocess_contigs(
    contig_path: str, contig_processed_path: str, seq_min_length: int = 2500
) -> None:
    """Process contigs and dump both a csv of labelled sequences and csv of labelled ids.

    Args:
        contig_path (str): metahit datapath
        contig_processed_path (str): Path to Processed Contig CSV
        seq_min_length (int): minimum length of sequence
    """
    if os.path.exists(contig_processed_path):
        print(f"CSV file already exists at {contig_processed_path}")
        return
    if os.path.exists(contig_path):
        print(f"Reading data from {contig_path}")
        with gzip.open(contig_path, "rt") as f:
            with open(contig_processed_path, "w") as contig_processed_csv:
                contig_processed_csv.write("id,sequence,species\n")
                short_seq = 0
                for i, contig in enumerate(SeqIO.parse(f, "fasta")):
                    seq = contig.seq
                    if len(str(seq)) < seq_min_length:
                        short_seq += 1
                        continue
                    species, _ = contig.id.split("_[")[-1].split("]_")
                    contig_processed_csv.write(f"{i},{str(seq)},{species}\n")
        print(f"Removed {short_seq} sequences that were shorter than {seq_min_length}.")
        return
    else:
        raise FileNotFoundError(f"File not found at {contig_path}")


def summary_stats(contig_processed_path: str) -> None:
    """Print summary statistics of the contigs dataset

    Args:
        data_path (str): Path to Contig CSV
    """
    contig_df = pd.read_csv(contig_processed_path)
    label_counts_series = contig_df["species"].value_counts()
    label_counts = label_counts_series.values
    print(
        f"Number of Contigs Total: {contig_df.shape[0]}\nNumber of Species: {len(label_counts)}"
    )
    print(
        f"Mininum Number of Contigs per Species: {min(label_counts)}\nMaximum Number of Contigs per Species: {max(label_counts)}\nMedian Number of Contigs per Species: {np.median(label_counts)}"
    )
    return


def sort_sequences(dna_sequences: list) -> tuple[list, np.array]:
    """Sorting sequences by length and returning sorted sequences and indices

    Args:
        data (list): List ID, DNA Sequence, Label from loaded CSV

    Returns:
        tuple[list, list]: Sorted DNA Sequences and corresponding indices
    """
    lengths = [len(seq) for seq in dna_sequences]
    idx_asc = np.argsort(lengths)
    idx_desc = idx_asc[::-1]
    dna_sequences = [dna_sequences[i] for i in idx_desc]

    return dna_sequences, idx_desc


def label_to_id(data: list[list]) -> tuple[np.array, dict]:
    """Convert labels to numeric values"""

    labels = [label[2] for label in data[1:]]
    sorted_set_labels = sorted(
        set(labels)
    )  # sort to ensure same ordering every time the fuhnction is run

    label2id = {label: i for i, label in enumerate(sorted_set_labels)}
    id2label = {i: label for label, i in label2id.items()}

    label_ids = np.array([label2id[l] for l in labels])
    return label_ids, id2label


def validate_input_array(array):
    "Returns array similar to input array but C-contiguous and with own data."
    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array)
    if not array.flags["OWNDATA"]:
        array = array.copy()

    assert array.flags["C_CONTIGUOUS"] and array.flags["OWNDATA"]

    return array


def get_threshold_dataset_indices(
    label_ids: list,
    threshold_dataset_indices_path: str,
    sample_rate=0.1,
) -> np.array:
    """Generates or loads a threshold dataset for calculating model-specific thresholds.

    This function samples a specified proportion of species, setting aside their contigs to form the threshold dataset.
    If the specified path exists, the function loads the dataset from it. Otherwise, it creates a new threshold dataset.

    Args:
        label_ids (list): List of species labels converted to IDs.
        threshold_dataset_indices_path (str): Path where the threshold dataset indices should be saved or loaded.
        sample_rate (float, optional): Proportion of species to include in the threshold dataset. Defaults to 0.1.

    Returns:
        np.array: An array of the same length as `label_ids`, with `1` for indices included in the threshold dataset,
                    and `0` for those not included.
    """

    if os.path.exists(threshold_dataset_indices_path):
        threshold_dataset_indices = np.load(threshold_dataset_indices_path)

        assert len(threshold_dataset_indices) == len(
            label_ids
        ), "Dimensions of loaded threshold indices and label_ids do not match"

        print(
            f"Loading threshold dataset indices\nThreshold dataset comprise {sum(threshold_dataset_indices)} contigs ({sum(threshold_dataset_indices)/len(threshold_dataset_indices)*100:.1f}%)\n"
        )

        return threshold_dataset_indices

    else:
        unique_label_ids = np.unique(label_ids)

        threshold_dataset_species_sampled = np.random.choice(
            unique_label_ids, int(len(unique_label_ids) * sample_rate), replace=False
        )

        threshold_dataset_indices = np.where(
            np.isin(label_ids, threshold_dataset_species_sampled), 1, 0
        )
        print(
            f"Sampling new threshold dataset indices\nThreshold dataset comprise {sum(threshold_dataset_indices)} contigs ({sum(threshold_dataset_indices)/len(threshold_dataset_indices)*100:.1f}%) "
        )

        np.save(threshold_dataset_indices_path, threshold_dataset_indices)

        return threshold_dataset_indices


def split_dataset(
    embeddings: np.array,
    label_ids: list,
    threshold_dataset_indices_path: str,
    sample_rate=0.1,
) -> tuple[np.array, np.array, np.array, np.array]:
    """Split the dataset into an evaluation dataset and a threshold dataset according to the indices returned by get_threshold_dataset_indices.

    Args:
        embeddings (np.array): Array of embeddings from the model.
        label_ids (list): List of species labels converted to IDs.
        threshold_dataset_indices_path (str): Path where the threshold dataset indices should be saved/loaded.
        sample_rate (float, optional): Proportion of species to use for the threshold dataset. Defaults to 0.1.

     Returns:
        tuple[np.array, np.array, np.array, np.array]: Tuple containing:
            - embeddings for evaluation,
            - embeddings for threshold calculation,
            - label IDs for evaluation,
            - label IDs for threshold calculation.

    """

    threshold_dataset_indices = get_threshold_dataset_indices(
        label_ids, threshold_dataset_indices_path, sample_rate
    )

    assert (
        len(threshold_dataset_indices) == len(embeddings) == len(label_ids)
    ), "Dimensions of threshold indices, embeddings, and label_ids do not match"

    embeddings_evaluate = embeddings[threshold_dataset_indices == 0]
    embeddings_threshold = embeddings[threshold_dataset_indices == 1]

    label_ids_evaluate = label_ids[threshold_dataset_indices == 0]
    label_ids_threshold = label_ids[threshold_dataset_indices == 1]

    assert len(embeddings_evaluate) + len(embeddings_threshold) == len(embeddings)
    assert len(label_ids_evaluate) + len(label_ids_threshold) == len(label_ids)

    return (
        embeddings_evaluate,
        embeddings_threshold,
        label_ids_evaluate,
        label_ids_threshold,
    )


def calculate_similarity_matrix(
    embeddings: np.array, min_similarity: float, output_file_path: str
) -> str:
    """Use pytables to store similarity matrix in HDF5 format and calculate similarity matrix
    in blocks to ease memory overhead. Inspiration for the solution is found here:
    https://medium.com/@ph_singer/handling-huge-matrices-in-python-dff4e31d4417.
    As we evaluate the similarities with the min_similarity within the calculation loop,
    the embeddings passed to the function must be normalized.


    Args:
        embeddings (np.array): IMPORTANT: Normalized embeddings
        min_similarity (float): distance min_similarity
        output_file_path (str): hd5 storage path

    Returns:
        str: Path to the stored similarity matrix
    """
    output_file_path = os.path.join("similarities", output_file_path)
    if os.path.exists(output_file_path):
        print(f"Similarity file already exists at {output_file_path}\n")
        return output_file_path

    else:
        print(f"Calculating similarities and storing in {output_file_path}")
        embeddings = embeddings.astype(np.float32)
        n = embeddings.shape[0]
        f = tb.open_file(output_file_path, "w")
        filters = tb.Filters(complevel=4, complib="blosc")
        similarities_h5 = f.create_carray(
            f.root, "similarities", tb.Float32Atom(), shape=(n, n), filters=filters
        )

        block_size = 100
        for i in tqdm.tqdm(
            range(0, n, block_size),
            desc=f"Computing Similarities",
        ):
            end_i = min(i + block_size, n)
            for j in range(0, n, block_size):
                end_j = min(j + block_size, n)
                block = np.dot(embeddings[i:end_i], embeddings[j:end_j].T)  # EE^T
                block[block < min_similarity] = 0  # Apply similarity threshold
                similarities_h5[i:end_i, j:end_j] = block  # Store the block
        f.close()

        return output_file_path
