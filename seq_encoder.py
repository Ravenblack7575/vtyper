import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import os
from tqdm import tqdm # Use standard tqdm
import argparse # Import the argparse module


# Nucleotide mapping

# Each nucleotide is represented as a 4-element vector.
# N (any base) or other unexpected characters will be encoded as all zeros.
NUCLEOTIDE_MAP = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1],
    'N': [0, 0, 0, 0] # Represent 'N' as all zeros
}
# Default encoding for any character not in NUCLEOTIDE_MAP (e.g., 'R', 'Y', etc.)
UNKNOWN_NUCLEOTIDE_ENCODING = [0, 0, 0, 0]


# Function to encode a single DNA sequence
def encode_sequence(sequence, nucleotide_map):
    encoded_sequence = []
    for nucleotide in sequence.upper():  # Convert to uppercase for case-insensitivity
        encoded_sequence.append(nucleotide_map.get(nucleotide, UNKNOWN_NUCLEOTIDE_ENCODING))
    return np.array(encoded_sequence)


# Function to load, encode, and label data
def load_and_encode_data(dataset_dir, nucleotide_map):
    all_sequences = []
    all_labels = []
    label_encoder = LabelEncoder()

    # Get the class names from the folder names
    # Filter out non-directory entries and sort for consistent order
    class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    print(f"Found classes: {class_names}")

    # Use standard tqdm for command line execution
    for class_name in tqdm(class_names, desc="Processing classes"):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                # Check for both .fasta and .fa extensions case-insensitively
                if filename.lower().endswith(('.fasta', '.fa')):
                    filepath = os.path.join(class_dir, filename)
                    try:
                        # Assuming each .fasta file contains one sequence
                        with open(filepath, 'r') as f:
                            sequence_lines = []
                            for line in f:
                                if not line.strip().startswith('>'): # Skip header lines
                                    sequence_lines.append(line.strip())
                            sequence = "".join(sequence_lines)

                        if sequence: # Only process non-empty sequences
                            encoded_seq = encode_sequence(sequence, nucleotide_map)
                            all_sequences.append(encoded_seq)
                            all_labels.append(class_name)
                    except Exception as e:
                        print(f"Error processing file {filepath}: {e}")

    # Convert labels to numerical format
    if not all_labels:
        print("No sequences found to encode.")
        return [], [], []

    encoded_labels = label_encoder.fit_transform(all_labels)
    # Convert numerical labels to one-hot encoding
    one_hot_labels = to_categorical(encoded_labels, num_classes=len(class_names))

    return all_sequences, one_hot_labels, label_encoder.classes_


# Function for padding the sequence
def seq_padding(sequences, max_sequence_length):
    if not sequences:
        return np.array([]) # Return empty array if no sequences

    # Determine the shape of the features based on the first sequence
    feature_dim = sequences[0].shape[1]
    padded_sequences = np.zeros((len(sequences), max_sequence_length, feature_dim), dtype=np.float32)

    for i, seq in enumerate(sequences):
        seq_length = min(seq.shape[0], max_sequence_length) # Ensure we don't exceed max_sequence_length
        padded_sequences[i, :seq_length, :] = seq[:seq_length, :] # Pad or truncate sequence

    return padded_sequences

# Define the path to save the processed data
def save_processed_data(padded_sequences, labels, class_names, save_path):
  # Create the save directory if it doesn't exist
  os.makedirs(save_path, exist_ok=True)
  save_path = save_path.rstrip('/')  # Remove trailing slash if present
  # Save the padded sequences and labels using numpy's save function
  np.save(os.path.join(save_path, 'padded_sequences.npy'), padded_sequences)
  np.save(os.path.join(save_path, 'labels.npy'), labels)
  np.save(os.path.join(save_path, 'class_names.npy'), class_names)

  print(f"Padded sequences saved to: {os.path.join(save_path, 'padded_sequences.npy')}")
  print(f"Labels saved to: {os.path.join(save_path, 'labels.npy')}")
  print(f"Class names saved to: {os.path.join(save_path, 'class_names.npy')}")


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Load, encode, pad, and save DNA sequence data.')
    parser.add_argument('dataset_dir', type=str, help='Path to the dataset directory containing class subdirectories.')
    parser.add_argument('--save_dir', type=str, default='./processed_data', help='Directory to save the processed data (default: ./processed_data)')

    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    save_directory = args.save_dir


    # Check if the dataset directory exists
    if not os.path.isdir(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found.")
        # Exit the script
        import sys
        sys.exit(1)
    else:
        sequences, labels, class_names = load_and_encode_data(dataset_dir, NUCLEOTIDE_MAP)

        if sequences:
            print(f"Loaded {len(sequences)} sequences")
            print(f"Loaded {len(labels)} labels")
            print(f"Class names: {class_names}")

            # Before padding, find the maximum sequence length
            # Find the maximum length among the loaded sequences
            max_sequence_length = max(len(seq) for seq in sequences) if sequences else 0
            print(f"Maximum sequence length found: {max_sequence_length}")

            # if you want to define a fixed maximum sequence length for padding instead
            # fixed_max_length = 1000 # Example fixed length
            # if max_sequence_length > fixed_max_length:
            #     print(f"Warning: Max sequence length ({max_sequence_length}) exceeds fixed padding length ({fixed_max_length}). Truncating sequences.")
            #     max_sequence_length = fixed_max_length # Use fixed length if shorter

            padded_sequences = seq_padding(sequences, max_sequence_length)
            print(f"Padded sequences shape: {padded_sequences.shape}")

            # Save the processed data
            save_processed_data(padded_sequences, labels, class_names, save_directory)

        else:
            print("No sequences were successfully loaded and processed.")


"""

How to Run with Command-Line Arguments:

## When you run this script from your terminal, you would provide the dataset directory path as the first argument:

python seq_encoder.py /path/to/your/dataset

## If you also want to specify a different save directory, you would use the --save_dir flag:

python seq_encoder.py /path/to/your/dataset --save_dir /

"""