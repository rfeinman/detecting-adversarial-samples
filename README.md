# Detecting Adversarial Samples from Artifacts
This repository contains the code for [Detecting Adversarial Samples from
Artifacts](https://arxiv.org/abs/1703.00410) (Feinman et al. 2017).

## Code Structure
The source code is located in the src/ subfolder, and scripts that users will
run to perform various steps are located in the scripts/ subfolder. An empty
subfolder, data/, is included for storing trained models and adversarial sample
arrays. Instructions for running the code are below.

## Running the Code
All of the scripts for running the various parts of the code are located
in the scripts/ subfolder.

### Train a new model
To train a new model for a particular data set, simply run

    python train_model.py -d=<dataset> -e=<nb_epochs>

where <dataset> is one of either mnist, cifar or svhn, and <nb_epochs>
is an integer indicating the number of epochs to train for. For example,
to train the MNIST model for 10 epochs, we would run

    python train_model.py -d=mnist -e=10

The model will be trained and saved into the data/ subfolder.

### Craft adversarial samples
todo

### Detect adversarial samples
todo
