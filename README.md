# VTyper
A simple CNN model for classification of flavivirus sequences.

### Packages required: 
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

Viral genomic sequences were downloaded from NCBI refseq and genbank databases (https://www.ncbi.nlm.nih.gov/labs/virus/vssi/#/). After a few attempts, I got a model that seems to work. This is a very simple experiment (and model), and I learned that I have so much more to learn.


MIT License
Copyright (c) 2025 Elizabeth A. S. Lim

