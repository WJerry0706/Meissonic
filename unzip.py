import pandas as pd
from PIL import Image
import io
import os
import csv
import json
from pathlib import Path

# === Configuration ===
input_folder = "parquets_father_dir"     # e.g., "./parquet_files"
output_image_dir = "output_images"
caption_format = "csv"                     # or "jsonl"
caption_output_file = f"captions.{caption_format}"

# === Setup ===
os.makedirs(output_image_dir, exist_ok=True)
image_counter = 0  # Global counter for unique image filenames

# Prepare caption output file
if caption_format == "csv":
    caption_file = open(caption_output_file, mode="w", newline='', encoding="utf-8")
    writer = csv.writer(caption_file)
    writer.writerow(["filename", "caption"])
elif caption_format == "jsonl":
    caption_file = open(caption_output_file, mode="w", encoding="utf-8")

# === Process all Parquet files ===
parquet_files = sorted(Path(input_folder).glob("*.parquet"))

for parquet_path in parquet_files:
    print(f"Processing: {parquet_path}")
    df = pd.read_parquet(parquet_path, engine="pyarrow")

    for _, row in df.iterrows():
        caption = row["caption"]
        img_bytes = row["image"]["bytes"]
        filename = f"image_{image_counter:05d}.png"
        image_path = os.path.join(output_image_dir, filename)

        # Save image
        image = Image.open(io.BytesIO(img_bytes))
        image.save(image_path)

        # Save caption
        if caption_format == "csv":
            writer.writerow([filename, caption])
        elif caption_format == "jsonl":
            json.dump({"image": filename, "caption": caption}, caption_file)
            caption_file.write("\n")

        image_counter += 1

# Close the caption file
caption_file.close()
print(f"✅ Done! Saved {image_counter} images and captions.")
