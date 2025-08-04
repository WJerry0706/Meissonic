import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import json

def count_tensor_frequencies_torch(directory_path, max_files_to_read=None):
    """
    Reads tensors from .pt files in a directory and counts the frequency
    of each integer using PyTorch. Can be limited to a max number of files.

    Args:
        directory_path (str): The path to the directory containing the .pt files.
        max_files_to_read (int, optional): The maximum number of files to read.
                                           If None, all files are read. Defaults to None.

    Returns:
        dict: A dictionary where keys are the integers and values are their counts.
    """
    all_elements = []
    print(f"Scanning for .pt files in '{directory_path}'...")

    # 1. Get a list of all .pt files first
    try:
        all_pt_files = [f for f in os.listdir(directory_path) if f.endswith(".pt")]
        # Sort the list to ensure we always process the same files in the same order
        all_pt_files.sort()
    except FileNotFoundError:
        print(f"Error: Directory not found at '{directory_path}'")
        return {}

    # 2. If a limit is specified, slice the list of files
    if max_files_to_read is not None:
        if max_files_to_read > len(all_pt_files):
            print(f"Warning: Requested {max_files_to_read} files, but only {len(all_pt_files)} found.")
        else:
            print(f"Limiting analysis to the first {max_files_to_read} of {len(all_pt_files)} sorted .pt files.")
        files_to_process = all_pt_files[:max_files_to_read]
    else:
        files_to_process = all_pt_files
    
    print(f"Starting analysis of {len(files_to_process)} .pt files...")
    
    # 3. Loop over the potentially limited list of files
    for filename in files_to_process:
        filepath = os.path.join(directory_path, filename)
        try:
            tensor = torch.load(filepath)
            all_elements.append(tensor.flatten())
        except Exception as e:
            print(f"Could not load tensor from {filepath}: {e}")

    if not all_elements:
        print("No tensors were successfully loaded.")
        return {}

    all_elements_tensor = torch.cat(all_elements)
    unique_elements, counts = torch.unique(all_elements_tensor, return_counts=True)
    frequency_dict = dict(zip(unique_elements.tolist(), counts.tolist()))
    
    print("\n--- Analysis Complete ---")
    return frequency_dict

def save_counts_to_file(counts_dict, output_file_path, format='txt'):
    """
    Saves the dictionary of codebook index counts to a file.
    """
    os.makedirs(os.path.dirname(output_file_path) or '.', exist_ok=True)

    if format == 'txt':
        with open(output_file_path, 'w') as f:
            f.write("Codebook Index Counts:\n")
            f.write("----------------------\n")
            sorted_items = sorted(counts_dict.items(), key=lambda item: item[0])
            for index, count in sorted_items:
                f.write(f"Index {index}: {count}\n")
        print(f"Counts saved to plain text file: {output_file_path}")
    elif format == 'json':
        with open(output_file_path, 'w') as f:
            json.dump(dict(counts_dict), f, indent=4)
        print(f"Counts saved to JSON file: {output_file_path}")
    else:
        print(f"Unsupported output format: {format}. Please choose 'txt' or 'json'.")

def plot_and_save_counts_by_index(counts_dict, output_image_path):
    """
    Plots the distribution of codebook indices in numerical order
    using a LINEAR scale for the y-axis.
    """
    if not counts_dict:
        print("No data to plot. Skipping image generation.")
        return

    indices = sorted(counts_dict.keys())
    counts = [counts_dict[index] for index in indices]
    
    os.makedirs(os.path.dirname(output_image_path) or '.', exist_ok=True)

    plt.figure(figsize=(20, 8))
    plt.bar(indices, counts, color='skyblue', width=1.0)
    # plt.yscale('log') # Removed as requested
    plt.xlabel('Codebook Index')
    plt.ylabel('Count') # Label updated
    plt.title('Distribution of Codebook Index Frequencies (Sorted by Index)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_image_path)
    print(f"Plot saved to: {output_image_path}")
    plt.close()

def plot_and_save_counts_by_frequency(counts_dict, output_image_path):
    """
    Plots the distribution of codebook indices sorted by frequency in descending order
    using a LINEAR scale for the y-axis.
    """
    if not counts_dict:
        print("No data to plot. Skipping image generation.")
        return

    sorted_items = sorted(counts_dict.items(), key=lambda item: item[1], reverse=True)
    counts = [item[1] for item in sorted_items]

    os.makedirs(os.path.dirname(output_image_path) or '.', exist_ok=True)

    plt.figure(figsize=(20, 8))
    plt.bar(range(len(counts)), counts, color='lightgreen')
    # plt.yscale('log') # Removed as requested
    plt.xlabel('Codebook Index (Ranked by Frequency)')
    plt.ylabel('Count') # Label updated
    plt.title('Distribution of Codebook Index Frequencies (Sorted by Count)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_image_path)
    print(f"Plot saved to: {output_image_path}")
    plt.close()

def plot_low_frequency_histogram(counts_dict, output_image_path, num_bins=50):
    """
    Selects the 98% of indices with the lowest frequencies and plots a
    histogram of their occurrence counts.
    """
    if not counts_dict:
        print("No data to plot. Skipping low-frequency histogram generation.")
        return
        
    # Sort items by count (frequency) in ascending order
    sorted_items = sorted(counts_dict.items(), key=lambda item: item[1])

    # Determine the cutoff for the bottom 98% of unique indices
    num_unique_indices = len(sorted_items)
    cutoff_index = int(num_unique_indices * 0.98)

    # Get only the counts of these low-frequency items
    low_freq_counts = [item[1] for item in sorted_items[:cutoff_index]]

    if not low_freq_counts:
        print("Not enough data to generate a low-frequency histogram.")
        return

    os.makedirs(os.path.dirname(output_image_path) or '.', exist_ok=True)
    
    plt.figure(figsize=(20, 8))
    # Create a histogram of the counts of the low-frequency tokens
    plt.hist(low_freq_counts, bins=num_bins, color='coral', edgecolor='black')
    
    plt.xlabel('Number of Occurrences (for a single token)')
    plt.ylabel('Number of Tokens in Bin')
    plt.title('Histogram of Counts for the 98% Least Frequent Tokens')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    try:
        plt.savefig(output_image_path)
        print(f"Low-frequency histogram saved to: {output_image_path}")
    except Exception as e:
        print(f"Error saving histogram to '{output_image_path}': {e}")
        
    plt.close()


# --- Main execution block ---

# --- Configuration ---
# 1. Directory containing your .pt tensor files
pt_files_directory = "token_imagenet"

# 2. Maximum number of files to process (e.g., 8000). Set to None for no limit.
FILE_LIMIT = 1500

# 3. Directory where all outputs will be saved
output_dir = "imagenet_results"
# --------------------

# --- Define Output Filenames ---
txt_output_file = os.path.join(output_dir, 'token_counts.txt')
image_output_file_by_index = os.path.join(output_dir, 'token_distribution_by_index_linear.png')
image_output_file_by_frequency = os.path.join(output_dir, 'token_distribution_by_frequency_linear.png')
low_freq_histogram_output_file = os.path.join(output_dir, 'low_frequency_histogram.png')

# Ensure the input directory exists
if not os.path.isdir(pt_files_directory):
    print(f"Error: Input directory '{pt_files_directory}' not found.")
    print("Please create it and add your .pt files, or update the 'pt_files_directory' variable.")
else:
    # Run the analysis with the specified file limit
    frequency_counts = count_tensor_frequencies_torch(pt_files_directory, max_files_to_read=FILE_LIMIT)
    
    # Process and save results only if data was processed
    if frequency_counts:
        print("\n--- Saving Results ---")
        save_counts_to_file(frequency_counts, txt_output_file, format='txt')
        
        # Generate and save all types of plots
        plot_and_save_counts_by_index(frequency_counts, image_output_file_by_index)
        plot_and_save_counts_by_frequency(frequency_counts, image_output_file_by_frequency)
        plot_low_frequency_histogram(frequency_counts, low_freq_histogram_output_file)
    
        print("\n--- Summary ---")
        print(f"Total unique indices found: {len(frequency_counts)}")
        
        # Calculate and display requested distribution information
        total_sum_counts = sum(frequency_counts.values())
        average_count = total_sum_counts / len(frequency_counts) if len(frequency_counts) > 0 else 0
    
        # Sort counts for top 10 and last 10
        sorted_counts = sorted(frequency_counts.items(), key=lambda item: item[1], reverse=True)
    
        top_10_indices = sorted_counts[:10]
        last_10_indices = sorted_counts[-10:]
    
        print("\n--- Distribution Information ---")
        print(f"Average count per index: {average_count:.2f}")
    
        print("\n--- Top 10 Most Frequent Indices ---")
        for index, count in top_10_indices:
            print(f"Index {index}: {count} occurrences")
    
        print("\n--- Last 10 Least Frequent Indices ---")
        for index, count in last_10_indices:
            print(f"Index {index}: {count} occurrences")
    else:
        print("\nNo data processed. Cannot provide summary or save results.")