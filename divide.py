import os
import shutil
from pathlib import Path

# === Configuration ===
image_folder = "output_images"         # Folder containing all images
num_parts = 1000                       # Number of folders to split into
prefix = "part_"                       # Folder name prefix

# Get list of image files (supporting common formats)
image_files = sorted([
    f for f in Path(image_folder).glob("*")
    if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
])
total_images = len(image_files)

# Calculate split sizes
images_per_part = total_images // num_parts
remainder = total_images % num_parts

print(f"Total images: {total_images}")
print(f"Images per folder: ~{images_per_part} (+{remainder} spread)")

# Split and move images
start_idx = 0
for part in range(num_parts):
    end_idx = start_idx + images_per_part + (1 if part < remainder else 0)
    part_folder = Path(image_folder) / f"{prefix}{part:04d}"
    part_folder.mkdir(exist_ok=True)

    for img_path in image_files[start_idx:end_idx]:
        shutil.move(str(img_path), part_folder / img_path.name)

    print(f"Moved {end_idx - start_idx} files to {part_folder}")
    start_idx = end_idx

print("✅ Done splitting images into 1000 folders.")
