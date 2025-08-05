# images_to_parquet.py
import os
import pathlib
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

"""
Converts a directory of images into a batched Parquet dataset.

This script recursively finds all images with specified formats in a source
directory and converts them into a dataset composed of multiple Parquet files.
Processing is done in batches to manage memory usage effectively.
"""

# --- Configuration ---
# 1. Set the path to your folder containing the images.
SOURCE_IMAGE_DIR = "output"

# 2. Set the path where the Parquet dataset will be saved.
OUTPUT_PARQUET_DIR = "My-Image-Parquet-Dataset"

# 3. Define the number of images to include in each Parquet file (batch).
BATCH_SIZE = 512

# 4. Specify the image formats you want to include.
IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']
# --------------------


def create_parquet_dataset_from_images():
    """Main function to discover images and create the Parquet dataset."""
    print("--- Starting Image to Parquet Conversion ---")

    # --- 1. Setup and File Discovery ---
    source_path = pathlib.Path(SOURCE_IMAGE_DIR)
    output_path = pathlib.Path(OUTPUT_PARQUET_DIR)
    output_path.mkdir(exist_ok=True)

    print(f"Searching for images in: {source_path.resolve()}")
    
    # Discover all image files recursively
    all_image_paths = []
    for fmt in IMAGE_FORMATS:
        all_image_paths.extend(list(source_path.glob(f'**/*{fmt}')))

    num_total_images = len(all_image_paths)

    if num_total_images == 0:
        print(f"Error: No images found in '{SOURCE_IMAGE_DIR}'. Please check the path and formats.")
        return

    print(f"Found {num_total_images} total images.")

    # --- 2. Define the Schema ---
    # This schema is similar to your example, with a placeholder for a caption/label
    # and a nested struct for the image bytes.
    schema = pa.schema([
        pa.field('caption', pa.string()),
        pa.field('image', pa.struct([
            pa.field('bytes', pa.binary())
        ]))
    ])

    # --- 3. Process in Batches ---
    num_batches = (num_total_images + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Processing {num_total_images} images in {num_batches} batches of size {BATCH_SIZE}...")

    for i in tqdm(range(num_batches), desc="Creating Parquet Files"):
        start_index = i * BATCH_SIZE
        end_index = min((i + 1) * BATCH_SIZE, num_total_images)
        batch_paths = all_image_paths[start_index:end_index]

        batch_data = []
        for image_path in batch_paths:
            try:
                # Read the raw bytes of the image file
                image_bytes = image_path.read_bytes()
                
                # The caption is empty because we don't have labels.
                # If you had labels, you would put them here.
                batch_data.append({
                    'caption': '',
                    'image': {'bytes': image_bytes}
                })
            except Exception as e:
                print(f"Warning: Skipping file {image_path} due to error: {e}")

        if not batch_data:
            continue

        # Create a PyArrow Table from the list of dictionaries
        table = pa.Table.from_pylist(batch_data, schema=schema)
        
        # Write the batch to a single Parquet file
        output_filename = f"train-{i:05d}-of-{num_batches:05d}.parquet"
        pq.write_table(
            table,
            os.path.join(OUTPUT_PARQUET_DIR, output_filename),
            compression='snappy'
        )

    print("\n--- Dataset Creation Complete! ---")
    print(f"Parquet dataset saved to: {output_path.resolve()}")


if __name__ == "__main__":
    # Create the source directory if it doesn't exist, so the script can run out-of-the-box.
    if not os.path.exists(SOURCE_IMAGE_DIR):
        print(f"Creating sample source directory '{SOURCE_IMAGE_DIR}'.")
        os.makedirs(SOURCE_IMAGE_DIR)
        print(f"Please add your images to the '{SOURCE_IMAGE_DIR}' directory and run the script again.")
    else:
        create_parquet_dataset_from_images()