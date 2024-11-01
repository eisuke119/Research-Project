import numpy as np
import transformers
import torch
import torch.utils.data as util_data
import torch.nn as nn
import tqdm
import os
from typing import List, Tuple


def calculate_tnf(
    dna_sequences: List[List[str]], kernel: bool = False
) -> Tuple[np.ndarray, int]:
    """Calculates tetranucleotide frequencies in a list of DNA sequences.

    This function computes the frequencies of all possible tetranucleotides (sequences of four nucleotides)
    within each DNA sequence in `dna_sequences`. Corresponds to calculate 4-mers. There are 4^4=256 combinations.

    Optionally, a kernel transformation can be applied to the resulting frequency embeddings.

    Args:
        dna_sequences (List[List[str]]): A list of DNA sequences, where each sequence is a list of nucleotide characters (A, T, C, or G).
        kernel (bool, optional): If True, applies a kernel transformation to the calculated frequencies using a pre-defined kernel matrix. Defaults to False.

    Returns:
        Tuple[np.ndarray, int]: A tuple containing:
            - embedding (np.ndarray): A 2D numpy array of shape (n, 256), where `n` is the number of DNA sequences,
              and each row contains the tetranucleotide frequency vector for a sequence. If `kernel` is True,
              the frequencies are transformed by the kernel matrix.
            - count_N (int): Total number of tetranucleotides with nucleotide N across all dna_sequences.
    """

    nucleotides = ["A", "T", "C", "G"]
    tetra_nucleotides = [
        a + b + c + d
        for a in nucleotides
        for b in nucleotides
        for c in nucleotides
        for d in nucleotides
    ]

    # Build mapping from tetra-nucleotide to index
    tnf_index = {tn: i for i, tn in enumerate(tetra_nucleotides)}

    # Build embeddings by counting TNFs
    embedding = np.zeros((len(dna_sequences), len(tetra_nucleotides)))
    count_N = 0
    for j, seq in enumerate(dna_sequences):
        for i in range(len(seq) - 3):
            try:
                tetra_nuc = seq[i : i + 4]
                embedding[j, tnf_index[tetra_nuc]] += 1
            except KeyError:  # there exist nucleotide N which will give error
                count_N += 1

    # Convert counts to frequencies
    total_counts = np.sum(embedding, axis=1)
    embedding = embedding / total_counts[:, None]

    return embedding, count_N


def calculate_dna2vec_embedding(dna_sequences: List[List[str]]) -> np.array:
    """
    Calculates the DNA2Vec embedding for a list of DNA sequences.

    The function then multiplies the TNF embedding with a 4-mer embedding matrix to obtain
    the DNA2Vec embedding. The 4-mer embedding matrix is pretrained embeddings on the hg38 (human genome) obtained from https://github.com/MAGICS-LAB/DNABERT_S/blob/main/evaluate/helper/4mer_embedding.npy.
    See more in paper dna2vec https://arxiv.org/abs/1701.06279.

    Args:
        dna_sequences (List[List[str]]): A list of DNA sequences, where each sequence is a list
                                         of nucleotide characters (A, T, C, or G).
    Returns:
        np.ndarray: A 2D numpy array representing the DNA2Vec embedding, where each row corresponds
                    to the embedding of a DNA sequence.
    """

    tnf_embeddings = os.path.join("embeddings", "tnf.npy")
    if os.path.exists(tnf_embeddings):
        print(f"Load TNF-embedding from file {tnf_embeddings}")
        tnf_embedding = np.load(tnf_embeddings)
    else:
        tnf_embedding, _ = calculate_tnf(dna_sequences)

    pretrained_4mer_embedding = np.load(
        "pretrained_4mer_embedding.npy"
    )  # dim (256,100)

    embedding = np.dot(tnf_embedding, pretrained_4mer_embedding)

    return embedding
