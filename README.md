# ml-optics-WACV2023

This repository contains code and some associated data from our WACV 2023 paper titled "". A link to the paper can be found here: <url>

The repository contains the following files

1. vae.py -- Python code used to train the 1D VAE used to generate pump patterns
2. gen_images.py and gen_database.py -- Python codes used to generate a training set of 1D profiles for the VAE. These codes generate 1D Bezier curves of 3840 pixels
3. ax_surrogate_model.py -- Python code used to perform Bayesian optimization on pump patterns, optimizing for directivity
4. nn_surrogate_for_expt.py -- Python code that uses some experimental data to create a surrogate model that the Bayesian optimization uses
5. nn.pth -- Example neural network surrogate model generated from the training code
