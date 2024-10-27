"""Skeleton for using custom dataloader and getting embeddings from model"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from data.data_loader import DNADataset


model = AutoModel.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-S")

store_csv_path = "path"
dataset = DNADataset(store_csv_path, tokenizer)


loader = DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=0
)  # ALWAYS: batch_size = 1, as input sequence length determines the paddings

for i, (inputs, _) in enumerate(loader):
    hidden_states = model(inputs)[0]
    embedding_mean = torch.mean(hidden_states[0], dim=0)
    # store embeddings if needed
    # torch.save(embedding_mean, f"../../../../../Downloads/METAHIT/data/metahit/embedding_mean_{i}.pt")
