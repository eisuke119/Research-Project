import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from sklearn.preprocessing import normalize
from PIL import Image
from tqdm import tqdm
import gc

def compute_and_save_heatmap_in_chunks(
    embeddings: np.ndarray,
    output_image: str,
    block_size: int = 1000
) -> None:
    """
    Computes the cosine similarity heatmap in chunks and saves the concatenated image vertically.

    Args:
        embeddings (np.ndarray): An array of shape (N, d) containing the normalized embeddings.
        output_image (str): The path where the final heatmap image will be saved.
        block_size (int): The size of the blocks to use during computation.
    """

    N = embeddings.shape[0]

    # Determine the number of chunks
    num_chunks = 10
    chunk_size = N // num_chunks + (N % num_chunks > 0)  # Ensure all rows are covered

    chunk_images = []  # List to store images of each chunk

    for chunk_idx in range(num_chunks):
        # Compute start and end indices for this chunk
        i_start = chunk_idx * chunk_size
        i_end = min(i_start + chunk_size, N)
        chunk_height = i_end - i_start

        # Initialize an array for this chunk's heatmap
        heatmap_chunk = np.zeros((chunk_height, N), dtype=np.float32)

        # Compute the heatmap for this chunk in blocks
        for j_start in tqdm(range(0, N, block_size), desc=f'Processing chunk {chunk_idx + 1}/{num_chunks}'):
            j_end = min(j_start + block_size, N)
            embeddings_j = embeddings[j_start:j_end]

            # Slice embeddings for the current chunk
            embeddings_i = embeddings[i_start:i_end]

            # Compute cosine similarities between the blocks
            similarities = np.dot(embeddings_i, embeddings_j.T)

            # Store the computed similarities in the heatmap chunk array
            heatmap_chunk[:, j_start:j_end] = similarities

        # Normalize similarities to [0, 1] for colormap mapping
        norm = Normalize(vmin=-1, vmax=1)
        heatmap_chunk = norm(heatmap_chunk)

        # Apply a colormap to map the normalized data to RGB values
        cmap = cm.get_cmap('viridis')  # You can choose other colormaps like 'hot', 'plasma', etc.
        heatmap_chunk = cmap(heatmap_chunk)  # Returns an (chunk_height, N, 4) array (RGBA)

        # Convert the RGBA values to 8-bit unsigned integers
        heatmap_chunk = (heatmap_chunk[:, :, :3] * 255).astype(np.uint8)  # Extract RGB channels

        # Create an image from the RGB array
        heatmap_chunk = Image.fromarray(heatmap_chunk, mode='RGB')
        chunk_images.append(heatmap_chunk)

        del heatmap_chunk
        gc.collect()

    # Concatenate the chunk images vertically
    total_width = chunk_images[0].width
    total_height = sum(img.height for img in chunk_images)
    concatenated_image = Image.new('RGB', (total_width, total_height))

    current_y = 0
    for img in chunk_images:
        concatenated_image.paste(img, (0, current_y))
        current_y += img.height

    # Save the concatenated image
    concatenated_image.save(output_image)

output_heatmap_image = 'heatmap.png'
block_size = 1000  # Adjust based on available memory
downsample_factor = 1  # Adjust to control the size of the output image

embedding = np.load('GROVER.npy')
embedding = normalize(embedding)

# Generate downsampled heatmap without storing the similarity matrix
print("Generating downsampled heatmap...")
#compute_and_save_heatmap_downsampled(embedding, output_heatmap_image, downsample_factor)
compute_and_save_heatmap_in_chunks(embedding, output_heatmap_image)