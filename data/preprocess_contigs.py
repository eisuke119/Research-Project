"""This file loads the dna sequences and dumps a csv of labelled sequences"""

import gzip
from Bio import SeqIO
import pandas as pd
import numpy as np


def preprocess_contigs(
    data_path: str, csv_path: str, seq_min_length: int = 2500
) -> None:
    """Process contigs and dump both a csv of labelled sequences and csv of labelled ids.

    Args:
        data_path (str): metahit datapath
        csv_path (str): Path to Contig CSV
        seq_min_length (int): minimum length of sequence
    """
    with gzip.open(data_path, "rt") as f:
        with open(csv_path, "w") as contig_store_csv:
            contig_store_csv.write("id,sequence,species\n")
            short_seq = 0
            for i, contig in enumerate(SeqIO.parse(f, "fasta")):
                seq = contig.seq
                if len(str(seq)) < seq_min_length:
                    short_seq += 1
                    continue
                try:
                    genome_id, scaffold = (
                        contig.id.split("|")[1],
                        contig.id.split("|")[3],
                    )
                    species, genome_positions = contig.id.split("_[")[-1].split("]_")
                    start_position, end_position = genome_positions.split(
                        "-"
                    )  # if needed
                    contig_store_csv.write(f"{i},{str(seq)},{species}\n")
                except:
                    raise ValueError(f"Could not extract metadata from ID: {contig.id}")

    print(f"Removed {short_seq} sequences that were shorter than {seq_min_length}.")
    return


def summary_stats(data_path: str) -> None:
    """Print summary statistics of the contigs dataset

    Args:
        data_path (str): Path to Contig CSV
    """
    contig_df = pd.read_csv(data_path)
    label_counts_series = contig_df["species"].value_counts()
    label_counts = label_counts_series.values
    print(
        f"Number of Contigs Total: {contig_df.shape[0]}\nNumber of Species: {len(label_counts)}"
    )
    print(
        f"Mininum Number of Contigs per Species: {min(label_counts)}\nMaximum Number of Contigs per Species: {max(label_counts)}\nMedian Number of Contigs per Species: {np.median(label_counts)}"
    )
    return


if __name__ == "__main__":
    path = "../../../../../Downloads/METAHIT/data/metahit/contigs.fna.gz"
    contig_csv_path = (
        "../../../../../Downloads/METAHIT/data/metahit/species_labelled_contigs.csv"
    )
    preprocess_contigs(path, contig_csv_path)
    summary_stats(contig_csv_path)
