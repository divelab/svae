# Spatial VAE via Matrix-Variate Normal Distributions

This is the tensorflow implementation of our recent work, "Spatial Variational Auto-Encoding via Matrix-Variate Normal Distributions". Please check the paper for details: https://arxiv.org/abs/1705.06821

In this work, we propose spatial VAEs that use latent variables as feature maps of larger size to explicitly capture spatial information. This is achieved by allowing the latent variables to be sampled from matrix-variate normal (MVN) distributions whose parameters are computed from the encoder network.

Experimental results:
1. CelebA dataset
