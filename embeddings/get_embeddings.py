"""Skeleton for using custom dataloader and getting embeddings from model"""

import csv
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from utils import DNADataset


if __name__ == "__main__":
    model = AutoModel.from_pretrained(
        "zhihan1996/DNABERT-S",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-S",
        model_max_length=20000,
        padding_side="right",
        trust_remote_code=True,
    )

    contig_csv_path = "../data/species_labelled_contigs.csv"
    with open(contig_csv_path) as csvfile:
        data = list(csv.reader(csvfile, delimiter=","))

    dataset = DNADataset(data)

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )  # ALWAYS: batch_size = 1, as input sequence length determines the paddings

    with open("../data/test.npy", "wb") as f:
        idx = []
        for i, (index, sequence) in enumerate(loader):
            idx.append(index)
            inputs = tokenizer(sequence, return_tensors="pt")["input_ids"]

            hidden_states = model(inputs)[0]
            embedding = torch.mean(hidden_states[0], dim=0).unsqueeze(0)
            # np.save(f, embedding.detach().numpy()) # Save embeddings as numpy array on the go
            if i == 0:
                embeddings = embedding
            else:
                embeddings = torch.cat(
                    (embeddings, embedding), dim=0
                )  # Concatenate embeddings and store all in one tensor
            if i == 2:
                break
        embeddings = np.array(embeddings.detach().cpu())
        np.save(f, embeddings)
