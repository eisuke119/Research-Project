"""Custom Dataloader for Contigs CSV file"""

from torch.utils.data import Dataset
import pandas as pd


class DNADataset(Dataset):
    def __init__(self, csv_file, tokenizer):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dna = self.data.iloc[idx]["sequence"]
        print("Length of DNA sequence", len(dna))
        label = self.data.iloc[idx]["species"]
        inputs = self.tokenizer(dna, return_tensors="pt")["input_ids"].squeeze(0)
        return inputs, label
