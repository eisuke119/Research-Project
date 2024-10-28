"""This file holds global variables used for model loading and representation extraction"""

MODELPATHS = [
    "zhihan1996/DNABERT-S",  # DNABERT-S
    "zhihan1996/DNABERT-2-117M",  # DNABERT-2-117M
    "zhihan1996/DNA_bert_6",  # DNABERT
    "LongSafari/hyenadna-large-1m-seqlen",  # HyenaDNA
    "PoetschLab/GROVER",  # GROVER
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",  # Nucleotide Transformer
    "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",  # Caduceus
    "togethercomputer/evo-1-8k-base",  # EVO
    "neuralbioinfo/prokbert-mini-long",  # ProkBERT
    "AIRI-Institute/gena-lm-bert-base-t2t",  # GenALM
]
TOKENIZERPATHS = [
    "zhihan1996/DNABERT-S",  # DNABERT-S
    "zhihan1996/DNABERT-2-117M",  # DNABERT-2-117M
    "zhihan1996/DNA_bert_6",  # DNABERT
    "LongSafari/hyenadna-large-1m-seqlen",  # HyenaDNA
    "PoetschLab/GROVER",  # GROVER
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",  # Nucleotide Transformer
    "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",  # Caduceus
    "togethercomputer/evo-1-8k-base",  # EVO
    "neuralbioinfo/prokbert-mini-long"  # ProkBERT
    "AIRI-Institute/gena-lm-bert-base-t2t",  # GenALM
]


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
