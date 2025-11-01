import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
import gradio as gr
import os # Import os to handle file paths

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

# Load the model
# Update the path to reflect the expected location on Hugging Face Spaces
# The model file should be placed in the root directory of your repository
model_path = "model5_flavi.keras"
model = keras.models.load_model(model_path)

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


article = "Note: I wanted to see if I could train a neural network to perform virus species typing. A quick review of\
    literature showed that using machine learning or deep learning approaches this way isn't new (Lopez-Rincon, 2021\
    DOI:10.1038/s41598-020-80363-5; Tampuu, 2019 DOI:10.1371/journal.pone.0222271). Performing classification using\
    a predictive model may be faster than the gold standard Blast search. This is because it wouldn't require\
    sequence alignment, which can sometimes take a long time. Dengue and Zika virus has been in the news\
    recently and are closely related. And so, I thought it might be fun to try to do one for Flaviviruses.\
    I reckon that a neural network with the right architecture could learn the genomic differences\
    between the Flaviviruses enough to correctly classify them. Viral genomic sequences are downloaded\
    from NCBI refseq and genbank databases (https://www.ncbi.nlm.nih.gov/labs/virus/vssi/#/). After\
    a few attempts, I got a model that seems to work.  Of course, this is a very simple experiment\
    (and model), and I have so much more to learn. But this little experiment was kinda fun to do\
    and play with."

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_virus_type,
    inputs=gr.Textbox(label="Enter cDNA Sequence"),
    outputs=[
        gr.Textbox(label="Predicted Virus Type"), # Output for the predicted type
        gr.Label(label="Prediction Confidence Scores") # Output for the colored bars
    ],
    title="Flavivirus cDNA Classifier",
    description="Enter a nucleotide sequence to determine see which Flavivirus the model classifies/predicts it is! (Classes: Dengue Virus Type 1-4, West Nile Virus, Japanese Encephalitis Virus and Zika Virus)"
)

# Launch the interface
iface.launch() # No debug=True for Hugging Face Spaces