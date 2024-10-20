import os
import gzip
import subprocess
from Bio import SeqIO


def read_fastq_gz(file_path):
    sequences = []
    total_records = 0

    # First pass to count total records
    with gzip.open(file_path, "rt") as handle:
        for _ in SeqIO.parse(handle, "fastq"):
            total_records += 1

    print(f"Total records to read: {total_records}")

    # Second pass to read the sequences and print progress
    with gzip.open(file_path, "rt") as handle:
        for i, record in enumerate(SeqIO.parse(handle, "fastq")):
            sequences.append(record)
            # Print progress every 10 records
            if (i + 1) % 10 == 0 or (i + 1) == total_records:
                print(
                    f"Processed {i + 1} of {total_records} records ({(i + 1) / total_records * 100:.2f}%)"
                )

    return sequences


if __name__ == "__main__":
    # Specify the URL of the FASTQ file
    url = (
        "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR646/009/SRR6468499/SRR6468499_1.fastq.gz"
    )

    # Specify the file name for the downloaded file
    file_path = "SRR6468499_1.fastq.gz"

    # Read the FASTQ file
    sequences = read_fastq_gz(file_path)

    if sequences:
        # Print the first sequence's information
        print(f"ID: {sequences[0].id}")
        print(f"Sequence: {sequences[0].seq}")
        print(f"Description: {sequences[0].description}")
        print(f"Quality Scores: {sequences[0].letter_annotations['phred_quality']}")

        # Print the length of each sequence
        print("\nLengths of all sequences:")
        for i, seq_record in enumerate(sequences):
            print(f"Sequence {i+1}: {len(seq_record.seq)} nucleotides")
    else:
        print("No sequences found in the file.")
