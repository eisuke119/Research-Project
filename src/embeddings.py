import warnings
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModel
import tqdm
import numpy as np

from .utils import sort_sequences
from .dataset import DNADataset

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", message="Increasing alibi size")


def get_available_device():
    """
    Returns the best available device for PyTorch computations.
    - If CUDA (GPU) is available, it returns 'cuda'.
    - If MPS (Apple GPU) is available, it returns 'mps'.
    - Otherwise, it returns 'cpu'.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_count = torch.cuda.device_count()
        return device, gpu_count
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        # returning to correctly handle batch size.
        return device, 1
    else:
        device = torch.device("cpu")
        # returning to correctly handle batch size.
        return device, 1


def get_embeddings(dna_sequences, model_name, model_path, save_path):

    embedding_dir = "embeddings"
    if os.path.exists(embedding_dir):
        save_path = os.path.join(embedding_dir, save_path)
        if os.path.exists(save_path):
            # Load already computed Embeddings
            print(f"Load embedding from file {save_path}")
            embeddings = np.load(save_path)

            if embeddings.shape[0] == len(dna_sequences):
                return embeddings
            else:
                print(
                    f"Mismatch in number of embeddings from {save_path} and DNA sequences.\nRecalculating embeddings."
                )
    if model_name == "TNF":
        embeddings = calculate_tnf(dna_sequences)
    elif model_name == "DNA2Vec":
        embeddings = calculate_dna2vec_embedding(dna_sequences, model_path)
    else:
        embeddings = calculate_llm_embedding(dna_sequences, model_path)

    with open(save_path, "wb") as f:
        np.save(f, embeddings)

    return embeddings


def calculate_llm_embedding(
    dna_sequences, model_path, model_max_length=None, batch_size=10
):
    # To reduce Padding overhead
    sorted_dna_sequences, idx = sort_sequences(dna_sequences)

    dna_sequences = DNADataset(sorted_dna_sequences)

    device, n_gpu = get_available_device()
    print(f"Using device: {device}\nwith {n_gpu} GPUs")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=model_max_length,
        padding_side="right",
        trust_remote_code=True,
        padding="max_length",
    )
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if n_gpu > 1:
        model = nn.DataParallel(model)

    model = model.to(device)

    data_loader = DataLoader(
        dna_sequences,
        batch_size=batch_size * n_gpu,
        shuffle=False,
        num_workers=2 * n_gpu,
    )
    for i, batch in enumerate(tqdm.tqdm(data_loader)):
        with torch.no_grad():
            inputs = tokenizer(batch, return_tensors="pt", padding=True)[
                "input_ids"
            ].to(device)
            hidden_states = model(inputs)[0]
            embedding = torch.mean(hidden_states[0], dim=0).unsqueeze(0)
            if i == 0:
                embeddings = embedding
            else:
                embeddings = torch.cat((embeddings, embedding), dim=0)

    embeddings = np.array(embeddings.detach().cpu())

    embeddings = embeddings[np.argsort(idx)]

    return embeddings


def calculate_tnf(
    dna_sequences: list[str], kernel: bool = False
) -> tuple[np.ndarray, int]:
    """Calculates tetranucleotide frequencies in a list of DNA sequences.

    This function computes the frequencies of all possible tetranucleotides (sequences of four nucleotides)
    within each DNA sequence in `dna_sequences`. Corresponds to calculate 4-mers. There are 4^4=256 combinations.

    Optionally, a kernel transformation can be applied to the resulting frequency embeddings.

    Args:
        dna_sequences (List[str]): A list of DNA sequences, where each sequence is a list of nucleotide characters (A, T, C, or G).
        kernel (bool, optional): If True, applies a kernel transformation to the calculated frequencies using a pre-defined kernel matrix. Defaults to False.

    Returns:
        Tuple[np.ndarray, int]: A tuple containing:
            - embeddings (np.ndarray): A 2D numpy array of shape (n, 256), where `n` is the number of DNA sequences,
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
    embeddings = np.zeros((len(dna_sequences), len(tetra_nucleotides)))
    no_missing_tns = 0
    for j, seq in tqdm.tqdm(enumerate(dna_sequences)):
        count_N = 0
        for i in range(len(seq) - 3):
            try:
                tetra_nuc = seq[i : i + 4]
                embeddings[j, tnf_index[tetra_nuc]] += 1
            except KeyError:  # there exist nucleotide N which will give error
                count_N += 1 / 4
        if len(seq) > 0:
            no_missing_tns += count_N / len(seq)

    print(f"Average Number of Missing Nucleotide (N): {count_N/len(dna_sequences)}")
    # Convert counts to frequencies
    total_counts = np.sum(embeddings, axis=1)
    embeddings = embeddings / total_counts[:, None]

    return embeddings


def calculate_dna2vec_embedding(dna_sequences: list[str], model_path: str) -> np.array:
    """
    Calculates the DNA2Vec embedding for a list of DNA sequences.

    The function then multiplies the TNF embedding with a 4-mer embedding matrix to obtain
    the DNA2Vec embedding. The 4-mer embedding matrix is pretrained embeddings on the hg38 (human genome) obtained from https://github.com/MAGICS-LAB/DNABERT_S/blob/main/evaluate/helper/4mer_embedding.npy.
    See more in paper dna2vec https://arxiv.org/abs/1701.06279.
    Args:
        dna_sequences (List[str]): A list of DNA sequences, where each sequence is a list
                                         of nucleotide characters (A, T, C, or G).
        model_path (str): Path to the model to be used for the embeddings.
    Returns:
        np.ndarray: A 2D numpy array representing the DNA2Vec embedding, where each row corresponds
                    to the embedding of a DNA sequence.
    """
    tnf_embeddings = os.path.join("embeddings", "TNF.npy")

    if os.path.exists(tnf_embeddings):
        print(f"Load TNF-embedding from file {tnf_embeddings}")
        tnf_embeddings = np.load(tnf_embeddings)
    else:
        tnf_embeddings = calculate_tnf(dna_sequences)

    pretrained_4mer_embedding = np.load(model_path)  # dim (256,100)
    embeddings = np.dot(tnf_embeddings, pretrained_4mer_embedding)

    return embeddings