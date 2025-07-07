import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import argparse

def preprocess_sequence_for_cnn(sequence_string, target_length=11520):
    """
    Preprocesses a single DNA sequence string for input into the CNN model,
    padding or truncating to a target length.
    Args:
        sequence_string (str): The DNA sequence string (e.g., "AAGTTG...").
        target_length (int): The desired length of the sequence after padding/truncation.
    Returns:
        np.ndarray: The preprocessed sequence as a numpy array,
                    reshaped to (1, target_length, 1, 4) for the CNN.
    """
    # Define the mapping for one-hot encoding
    nucleotide_map = {'A': [1, 0, 0, 0],
                      'C': [0, 1, 0, 0],
                      'G': [0, 0, 1, 0],
                      'T': [0, 0, 0, 1],
                      'N': [0, 0, 0, 0]} # Handle 'N' as all zeros

    # Pad or truncate the sequence to the target length
    if len(sequence_string) < target_length:
        # Pad with 'N' to the target length
        padded_sequence = sequence_string + 'N' * (target_length - len(sequence_string))
    elif len(sequence_string) > target_length:
        # Truncate the sequence
        padded_sequence = sequence_string[:target_length]
    else:
        # Sequence is already the target length
        padded_sequence = sequence_string

    # Convert the padded/truncated sequence string to a list of one-hot encoded vectors
    encoded_sequence = [nucleotide_map.get(base.upper(), [0, 0, 0, 0]) for base in padded_sequence] # Handle unknown bases

    # Convert the list to a numpy array
    encoded_sequence_array = np.array(encoded_sequence, dtype=np.float32)

    # Reshape for the CNN: (1, target_length, 1, 4)
    # Add a batch dimension (1), a width dimension (1), and keep height (target_length) and channels (4)
    preprocessed_input = np.expand_dims(encoded_sequence_array, axis=0) # Add batch dimension
    preprocessed_input = np.expand_dims(preprocessed_input, axis=2) # Add width dimension

    return preprocessed_input

# Define and fit the label encoder with the class names your model was trained on
# Replace the placeholder list with the actual list of virus types
label_encoder = LabelEncoder()
# Assuming your model was trained on 7 classes, replace with your actual class names
label_encoder.fit(['Dengue Virus Type 1', 'Dengue Virus Type 2', 'Dengue Virus Type 3', 'Dengue Virus Type 4', 'Japanese Encephalitis Virus', 'West Nile Virus', 'Zika Virus'])


def predict_virus_type(sequence_string):
    # Clean the input sequence string: remove whitespace and newline characters
    cleaned_sequence_string = "".join(sequence_string.split())

    # Preprocess the input sequence
    target_sequence_length = 11520
    preprocessed_sequence = preprocess_sequence_for_cnn(cleaned_sequence_string, target_length=target_sequence_length)

    # Get prediction from the model
    prediction = model.predict(preprocessed_sequence)[0] # Get the prediction for the single input

    # Get the predicted class index and convert to virus type
    predicted_class_index = np.argmax(prediction)
    predicted_virus_type = label_encoder.classes_[predicted_class_index]

    # Format the output as a dictionary for gr.Label
    confidence_scores = {}
    for i, class_name in enumerate(label_encoder.classes_):
        confidence_scores[class_name] = float(prediction[i]) # Convert numpy float to Python float

    return predicted_virus_type, confidence_scores # Return both the predicted type and the confidence scores dictionary


# Main function to run the prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict virus type from DNA sequence using a trained model.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained Keras model file (.keras)")
    parser.add_argument("--fasta", type=str, help="Path to a FASTA file containing the DNA sequence")
    args = parser.parse_args()

    # Load the model from the user-specified path
    model = keras.models.load_model(args.model)

    # Get sequence from FASTA file or user input
    if args.fasta:
        # Read the sequence from the FASTA file (skip header lines)
        with open(args.fasta, "r") as f:
            sequence_lines = [line.strip() for line in f if not line.startswith(">")]
        user_sequence = "".join(sequence_lines)
        print(f"Loaded sequence from {args.fasta}")
    else:
        user_sequence = input("Enter DNA sequence: ").strip()

    if not user_sequence:
        print("No sequence entered.")
    else:
        predicted_type, confidence_scores = predict_virus_type(user_sequence)
        print(f"Predicted Virus Type: {predicted_type}")
        print("Confidence Scores:")
        for virus_type, score in confidence_scores.items():
            print(f"{virus_type}: {score:.4f}")

"""
# Example usage:

For FASTA file:
python [vtyper_predict.py] --model 'path/to/model.keras' --fasta 'path/to/sequence.fasta'
For manual input:
python [vtyper_predict.py] 'path/to/model.keras'
"""
