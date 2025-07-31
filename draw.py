import torch
import os
import matplotlib.pyplot as plt

def count_tensor_frequencies_torch(directory_path):
    """
    Reads tensors from .pt files in a directory and counts the frequency
    of each integer using PyTorch.

    Args:
        directory_path (str): The path to the directory containing the .pt files.

    Returns:
        dict: A dictionary where keys are the integers and values are their counts.
    """
    all_elements = []

    # 1. Read all tensors from .pt files and collect their elements
    for filename in os.listdir(directory_path):
        if filename.endswith(".pt"):
            filepath = os.path.join(directory_path, filename)
            try:
                # Load the tensor
                tensor = torch.load(filepath)
                # Flatten the tensor and append
                all_elements.append(tensor.flatten())
            except Exception as e:
                print(f"Could not load tensor from {filepath}: {e}")

    # Handle the case where no .pt files were found
    if not all_elements:
        print("No .pt files found in the specified directory.")
        return {}

    # 2. Concatenate all elements into a single tensor
    all_elements_tensor = torch.cat(all_elements)

    # 3. Use PyTorch to count unique elements and their frequencies
    unique_elements, counts = torch.unique(all_elements_tensor, return_counts=True)

    # 4. Convert the results to a dictionary
    frequency_dict = dict(zip(unique_elements.tolist(), counts.tolist()))

    return frequency_dict

# --- Example Usage ---
# Replace this with the actual path to your .pt files
pt_files_directory = "token_output"
plot_output_directory = "visualizations"
plot_filename = "frequency_plot.png"

# Ensure the output directories exist
os.makedirs(pt_files_directory, exist_ok=True)
os.makedirs(plot_output_directory, exist_ok=True)


# Run the function
frequency_counts = count_tensor_frequencies_torch(pt_files_directory)

# Print the result
print("\nFinal Frequency Dictionary:")
print(frequency_counts)

# --- Visualization ---
if frequency_counts:
    indices = list(frequency_counts.keys())
    frequencies = list(frequency_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(indices, frequencies, color='skyblue')
    plt.xlabel("Index")
    plt.ylabel("Frequency")
    plt.title("Frequency of Indices")
    plt.xticks(indices)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_filepath = os.path.join(plot_output_directory, plot_filename)
    plt.savefig(plot_filepath)
    print(f"\nPlot saved as '{plot_filepath}'")
else:
    print("Could not generate a plot because the frequency dictionary is empty.")