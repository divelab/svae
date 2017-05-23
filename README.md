# Spatial VAE via Matrix-Variate Normal Distributions

This is the tensorflow implementation of our recent work, "Spatial Variational Auto-Encoding via Matrix-Variate Normal Distributions". Please check the [paper](https://arxiv.org/abs/1705.06821) for details.

In this work, we propose spatial VAEs that use latent variables as feature maps of larger size to explicitly capture spatial information. This is achieved by allowing the latent variables to be sampled from matrix-variate normal (MVN) distributions whose parameters are computed from the encoder network.

## Experimental results:
1. CelebA dataset

![image](https://github.com/divelab/Spatial-VAE-via-MVND/blob/master/celeba_new.png)

2. Cifar dataset

![image](https://github.com/divelab/Spatial-VAE-via-MVND/blob/master/cifar_new.png)


In both figures above, the first and second rows shows training images and images generated by the original VAEs. The
remaining three rows are the results of naïve spatial VAEs, spatial VAEs via MVN distributions and
spatial VAEs via low-rank MVN distributions, respectively.


## How to use the code:



