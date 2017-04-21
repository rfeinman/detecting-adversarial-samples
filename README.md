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

    python train_model.py -d=<data_set> -e=<nb_epochs>

where <data_set> is one of either 'mnist,' 'cifar' or 'svhn,' and <nb_epochs>
is an integer indicating the number of epochs to train for. For example,
to train the MNIST model for 10 epochs, we would run

    python train_model.py -d=mnist -e=10

The model will be trained and saved into the data/ subfolder and named
model_<data_set>.h5.

### Craft adversarial samples
To craft adversarial samples for a particular data set, you must first
train the model for that data set (details above). Then, simply run

    python craft_adv_samples.py -d=<data_set> -a=<attack>

where <data_set> is the same as above and `<attack>` is one of either 'fgsm,'
'jsma,' 'bim-a,' 'bim-b' or 'all,' indicating which method to use to craft
adversarial samples. For example, to craft adversarial samples for the
MNIST model using fgsm, we would run

    python craft_adv_samples.py -d=mnist -a=fgsm

If 'all' is chosen (the default), all types of adversarial samples will be
generated. Arrays holding the adversarial samples are stored in the data/
subfolder and named X_adv_`<attack>`.npy.

### Detect adversarial samples
To run the detection script, you must first train the model and craft
adversarial samples for each data set you would like to use (details above).
Then, simply run

    python detect_adv_samples.py -d=<data_set> -a=<attack>

where <data_set> and `<attack>` are the same as described above. For all of the
adversarial samples provided, an equal number of noisy samples will be generated
and included alongside the original samples as part of the 'negative' class
for the detector. The perturbation size of these noisy samples is determined
based on the average L2 perturbation size of the adversarial samples. Then,
the Bayesian uncertainty and kernel density features will be computed for each
of the normal, noisy and adversarial samples. A logistic regression model is
trained on these features and the detector is built.
