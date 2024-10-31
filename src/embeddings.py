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


def get_embeddings(dna_sequences, model_path, save_path):

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
    # if model
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
        if i == 2:
            break

    embeddings = np.array(embeddings.detach().cpu())

    embeddings = embeddings[np.argsort(idx)]

    return embeddings
