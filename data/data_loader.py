"""Custom Dataloader for Contigs CSV file"""

from torch.utils.data import Dataset
import pandas as pd


class DNADataset(Dataset):
    """Custom Dataset for DNA Sequences

    Args:
        Dataset (Dataset): Pytorch Dataset
    """

    def __init__(self, csv_file: str, tokenizer: object):
        """_summary_

        Args:
            csv_file (str): Filepath to contig csv
            tokenizer (object): Model Tokenizer
        """
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Get length of the dataset

        Returns:
            int: N
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[object, str]:
        """Retrieve item from dataset

        Args:
            idx (int): _description_

        Returns:
            tuple[object, str]: DNA embeddings, Species
        """

        dna = self.data.iloc[idx]["sequence"]
        label = self.data.iloc[idx]["species"]
        inputs = self.tokenizer(dna, return_tensors="pt")["input_ids"].squeeze(0)
        return inputs, label
