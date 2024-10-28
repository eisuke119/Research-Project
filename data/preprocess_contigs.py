"""This file loads the dna sequences and dumps a csv of labelled sequences"""

import gzip
from Bio import SeqIO


def preprocess_contigs(data_path: str, csv_path: str, id_label_csv_path: str) -> None:
    """Process contigs and dump both a csv of labelled sequences and csv of labelled ids.

    Args:
        data_path (str): metahit datapath
        csv_path (str): csv path to labelled contigs
        id_label_csv_path (str): csv path to labelled ids
    """
    with gzip.open(data_path, "rt") as f:
        with open(csv_path, "w") as store_csv:
            store_csv.write("id,sequence,species\n")
            with open(id_label_csv_path, "w") as id_to_species:
                id_to_species.write("id,species\n")
                for i, contig in enumerate(SeqIO.parse(f, "fasta")):
                    seq = contig.seq
                    genome_id, scaffold = (
                        contig.id.split("|")[1],
                        contig.id.split("|")[3],
                    )  # if needed
                    species, genome_positions = contig.id.split("_[")[-1].split("]_")
                    start_position, end_position = genome_positions.split(
                        "-"
                    )  # if needed
                    store_csv.write(f"{i},{str(seq)},{species}\n")
                    id_to_species.write(f"{i},{species}\n")
    return


if __name__ == "__main__":
    path = "path"
    store_csv_path = "path"
    id_to_species_path = "path"
    preprocess_contigs(path, store_csv_path, id_to_species_path)
