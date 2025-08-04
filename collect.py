import os
from pathlib import Path

# === Configuration ===
# The folder where the .jpg files are located.
parent_folder = "output_images"

def delete_specific_jpgs(target_dir):
    """
    Finds and deletes JPG/JPEG files in a directory whose filenames
    end with '_1' (e.g., 'image_1.jpg').

    Args:
        target_dir (str): The path to the directory to scan.
    """
    target_path = Path(target_dir)

    # Safety check to ensure the target folder exists
    if not target_path.is_dir():
        print(f"❌ Error: The specified folder '{target_dir}' does not exist.")
        return

    print(f"Scanning '{target_path}' for .jpg and .jpeg files ending with '_1'...")

    # Find all files whose name (stem) ends with '_1' and extension is .jpg or .jpeg
    files_to_delete = []
    for f in target_path.iterdir():
        # Check if it's a file and the stem ends with '_1'
        if f.is_file() and f.stem.endswith('_1'):
            # Check if the extension is .jpg or .jpeg (case-insensitive)
            if f.suffix.lower() in ['.png', '.jpeg']:
                files_to_delete.append(f)

    # --- Confirmation and Deletion Step ---
    if not files_to_delete:
        print("✅ No matching files found to delete.")
        return

    print(f"\nFound {len(files_to_delete)} files to be permanently deleted:")
    for file_path in files_to_delete:
        print(f"  - {file_path.name}")

    # Ask for user confirmation
    try:
        confirm = input("\nARE YOU SURE you want to permanently delete these files? (y/n): ").lower()
    except KeyboardInterrupt:
        print("\n❌ Deletion cancelled by user.")
        return
        
    if confirm == 'y':
        print("Proceeding with deletion...")
        deleted_count = 0
        for file_path in files_to_delete:
            try:
                file_path.unlink()  # This permanently deletes the file
                print(f"  - Deleted: {file_path.name}")
                deleted_count += 1
            except Exception as e:
                print(f"  - ❌ Error deleting {file_path.name}: {e}")
        
        print(f"\n✅ Successfully deleted {deleted_count} files.")
    else:
        print("❌ Deletion aborted by user. No files were changed.")


# --- Run the script ---
if __name__ == "__main__":
    delete_specific_jpgs(parent_folder)