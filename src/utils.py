import gzip
import os
import tables as tb

import tqdm
import numpy as np
from Bio import SeqIO
import pandas as pd


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

    label2id = {label: i for i, label in enumerate(set(labels))}
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


def calculate_similarity_matrix(
    embeddings: np.array, min_similarity: float, output_file_path: str
) -> None:
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
        None
    """
    output_file_path = os.path.join("similarities", output_file_path)
    embeddings = embeddings.astype(np.float32)
    n = embeddings.shape[0]
    f = tb.open_file(output_file_path, "w")
    filters = tb.Filters(complevel=4, complib="blosc")
    similarities_h5 = f.create_carray(
        f.root, "similarities", tb.Float32Atom(), shape=(n, n), filters=filters
    )

    block_size = 10
    for i in tqdm.tqdm(
        range(0, n, block_size), desc=f"Computing Similarities from {i}-{i+block_size}"
    ):
        end_i = min(i + block_size, n)
        for j in range(0, n, block_size):
            end_j = min(j + block_size, n)
            block = np.dot(embeddings[i:end_i], embeddings[j:end_j].T)  # EE^T
            block[block < min_similarity] = 0  # Apply similarity threshold
            similarities_h5[i:end_i, j:end_j] = block  # Store the block
    f.close()

    return
