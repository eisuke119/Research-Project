"""Custom Dataloader for Contigs CSV file"""

from torch.utils.data import Dataset


class DNADataset(Dataset):
    """Custom Dataset for DNA Sequences

    Args:
        Dataset (Dataset): Pytorch Dataset
    """

    def __init__(self, data: list) -> None:
        """_summary_

        Args:
            csv_file (str): Filepath to contig csv
            tokenizer (object): Model Tokenizer
        """
        self.sequence = [i[1] for i in data[1:]]
        self.id = [i[0] for i in data[1:]]

    def __len__(self) -> int:
        """Get length of the dataset

        Returns:
            int: N
        """
        return len(self.sequence)

    def __getitem__(self, idx: int) -> tuple[int, str]:
        """Retrieve item from dataset

        Args:
            idx (int): index of the item

        Returns:
            str: DNA Sequence
        """
        index = self.id[idx]
        sequence = self.sequence[idx]
        return index, sequence
