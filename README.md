# VTyper
A simple CNN model for classification of flavivirus sequences.



#### Objectives

The primary **objective** of this project was to **train and implement a model capable of predicting or classifying viral species based on a given input cDNA or genomic sequence**. Specifically, a deep learning model was built to classify **seven species of flavivirus**. The successful outcome offers a **faster alternative to computationally intensive methods like Blast** for applications such as **batch screening and sorting of contigs or assembled sequences**. This approach leverages the power of deep learning to identify unique patterns in genomic sequences, treating them similarly to how images are processed.

#### Methods

The classification model was built using a **basic convolutional neural network (CNN) architecture**. Sequences from the seven target flavivirus species (including Dengue Types 1 through 4, Zika Virus, West Nile Virus, and Japanese Encephalitis Virus) were downloaded from **NCBI Refseq and Genbank**. To prepare the sequences for the CNN, they were **vectorized using one-hot encoding**, which maps nucleotides (A, C, G, T) to unique binary vectors. A dedicated **final test set was set aside** before training commenced to prevent data contamination. The final simplified model architecture featured a single convolutional 2D layer, a MaxPooling2D layer, and dense layers culminating in a SoftMax output layer.

#### Results

The **final simplified model achieved a test accuracy of 97.8%** (Test Loss: 0.0995), demonstrating an improvement over the initial design. When evaluated using the completely **unseen final test set (160 predictions), only 7 predictions did not match the true labels**. Crucially, the model was effective in **detecting a possible misidentified sequence deposited in Genbank**. One sequence (MZ284953.1) labeled as Dengue virus Type 3 in the Genbank database was predicted by the model as **Dengue virus Type 1 with 99% confidence**. This prediction was subsequently confirmed through a manual check using Blastn, validating the utility of the CNN model for quick screening and **finding errors in public databases**.

(This project summary was generated using NotebookLM)

=======================================================================


### Libraries / Packages required: 
- Python 3.12
- TensorFlow/Keras 3.10
- numpy
- scikit-learn
- tqdm

Demo: https://huggingface.co/spaces/Ravenblack7575/vtyper

### Running this on WSL2 / Ubuntu 24.04

1) The script vtyper_predict.py can run a classification/prediction on your local machine if you have wsl or Python environment installed. It can take a fasta file or DNA sequence as input.
2) It outputs the most likely flavivirus species, along with prediction scores.

Example use:

For FASTA file:

python vtyper_predict.py --model 'path/to/model.keras' --fasta 'path/to/sequence.fasta'

For manual input:

python vtyper_predict.py --model 'path/to/model.keras'




### Motivation (June 2025)
Motivation/Notes: I wanted to see if I could train a neural network to perform virus species typing. A quick review of literature showed that using machine learning or deep learning approaches this way isn't new (Lopez-Rincon, 2021 DOI:10.1038/s41598-020-80363-5; Tampuu, 2019 DOI:10.1371/journal.pone.0222271). One observation is performing classification using a predictive model is faster than the gold standard Blast search (but would never be as accurate as Blast which involves direct alignment and comparison of sequences). But it could be good enough for quick screening if one has a large pool of contigs or sequences to screen.

Viral genomic sequences were downloaded from NCBI refseq and genbank databases (https://www.ncbi.nlm.nih.gov/labs/virus/vssi/#/). This is a very simple experiment (and model).


MIT License
Copyright (c) 2025 Elizabeth A. S. Lim

