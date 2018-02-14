# Detecting Adversarial Samples from Artifacts
This repository contains the code for the paper [Detecting
Adversarial Samples from Artifacts](https://arxiv.org/abs/1703.00410)
(Feinman et al., 2017).

## Requirements & Setup
This code repository requires Keras > 2.0 and TensorFlow. Keras must be
configured to use TensorFlow backend. A full list of requirements can be found
in `requirements.txt`. To install, run the following command to clone the
repository into a folder of your choice:

    git clone https://github.com/rfeinman/detecting-adversarial-samples.git

On UNIX machines, after cloning this repository, it is
recommended that you add the path to the repository to your `PYTHONPATH`
environment variable to enable imports from any folder:

    export PYTHONPATH="/path/to/detecting-adversarial-samples:$PYTHONPATH"


## Code Structure
The source code is located in the detect/ subfolder, and scripts that users will
run to perform various steps are located in the scripts/ subfolder. An empty
subfolder, data/, is included for storing trained models and adversarial sample
arrays. Instructions for running the code are below.

## Running the Code
All of the scripts for running the various parts of the code are located
in the scripts/ subfolder.

### 1. Train a new model
To train a new model for a particular data set, simply run

    python train_model.py -d=<dataset> -e=<nb_epochs>

where `<dataset>` is one of either 'mnist,' 'cifar' or 'svhn,' and `<nb_epochs>`
is an integer indicating the number of epochs to train for. We recommend using
10 epochs for MNIST, and 60 for each of CIFAR and SVHN. For example, to train
the MNIST model for 10 epochs, we would run

    python train_model.py -d=mnist -e=10

The model will be trained and saved into the data/ subfolder and named
`model_<dataset>.h5`. An optional batch size parameter is also available,
specified with `-b=<batch_size>`. The default training batch size is 128.

### 2. Craft adversarial samples
To craft adversarial samples for a particular data set, you must first
train the model for that data set (details above). Then, simply run

    python craft_adv_samples.py -d=<dataset> -a=<attack>

where `<dataset>` is the same as above and `<attack>` is one of either 'fgsm,'
'jsma,' 'bim-a,' 'bim-b' or 'all,' indicating which method to use to craft
adversarial samples. For example, to craft adversarial samples for the
MNIST model using FGSM, we would run

    python craft_adv_samples.py -d=mnist -a=fgsm

If 'all' is chosen (the default), all types of adversarial samples will be
generated. Arrays holding the adversarial samples are stored in the data/
subfolder and named `Adv_<dataset>_<attack>.npy`. An optional batch size
parameter for evaluating adversarial samples is again provided
(`-b=<batch_size>`). The default is 256.

### 3. Detect adversarial samples
To run the detection script, you must first train the model and craft
adversarial samples for each data set you would like to use (details above).
Then, simply run

    python detect_adv_samples.py -d=<dataset> -a=<attack>

where `<dataset>` and `<attack>` are the same as described above. An optional
batch size parameter is again provided (`-b=<batch_size>`). For all of the
adversarial samples provided, an equal number of noisy samples will be generated
and included alongside the original samples as part of the 'negative' class
for the detector. The perturbation size of these noisy samples is determined
based on the average L2 perturbation size of the adversarial samples. Then,
the Bayesian uncertainty and kernel density features will be computed for each
of the normal, noisy and adversarial samples. A logistic regression model is
trained on these features and the detector is built.

## MNIST Demonstration
Here, a simple demonstration is provided of the commands issued to run the full
experiment with MNIST, using the FGSM attack. The following commands are used
to run all 3 steps:

    1. python train_model.py -d=mnist -e=10
    2. python craft_adv_samples.py -d=mnist -a=fgsm
    3. python detect_adv_samples.py -d=mnist -a=fgsm
