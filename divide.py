import os
import shutil
from pathlib import Path

# === Configuration ===
image_folder = "output"         # Folder containing all original images
destination_folder = "split_images"    # New folder to store the divided image folders
num_parts = 1000                       # Number of folders to split into
prefix = "part_"                       # Folder name prefix

# Create the main destination folder if it doesn't exist
Path(destination_folder).mkdir(exist_ok=True)

# Get list of image files (supporting common formats)
print(f"Scanning for images in '{image_folder}'...")
image_files = sorted([
    f for f in Path(image_folder).glob("*")
    if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
])
total_images = len(image_files)

if total_images == 0:
    print("❌ No images found in the source folder. Exiting.")
    exit()

# Calculate split sizes
images_per_part = total_images // num_parts
remainder = total_images % num_parts

print(f"Total images: {total_images}")
print(f"Destination folder: '{destination_folder}'")
print(f"Images per folder: ~{images_per_part} (+{remainder} spread)")

# Split and copy images
start_idx = 0
for part in range(num_parts):
    # Determine the slice of files for the current part
    end_idx = start_idx + images_per_part + (1 if part < remainder else 0)
    
    # Define the path for the new subfolder (e.g., "split_images/part_0000")
    part_folder = Path(destination_folder) / f"{prefix}{part:04d}"
    part_folder.mkdir(exist_ok=True)

    # Get the image files for the current part
    files_to_copy = image_files[start_idx:end_idx]

    # Copy files instead of moving them
    for img_path in files_to_copy:
        shutil.copy(str(img_path), part_folder / img_path.name)

    print(f"Copied {len(files_to_copy)} files to {part_folder}")
    start_idx = end_idx

print(f"\n✅ Done copying images into {num_parts} folders inside '{destination_folder}'.")
print("Original image folder remains unchanged.")