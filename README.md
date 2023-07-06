# autoencoder

My implementation for a simple CNN-based autoencoder for 2D image data.

**Requirements**

Python 3.7 or greater, along the following libraries:

Numpy
Matplotlib
PyTorch

**Example**

The files driver.py and example_auto2D.py illustrate using autoencoder to learn a 2-D representation of a 
128x128 images of Gaussian distributions. Elements in the dataset differ only by their mean position, which 
is chosen at random from a circle. Thus, the optimal representation of this data in 2-D latent space 
is a circle (representing the position). It can be seen in the results that the autoencoder successfully learns 
a topologically close approximation of this.

![](https://github.com/adegenna/autoencoder/blob/master/figs/gauss.png)
*Example application of autoencoder to 128x128 pixelated images of Gaussian data*

![](https://github.com/adegenna/autoencoder/blob/master/figs/trainerror.png)
*Average mean-squared error on training set w.r.t. epoch*