import numpy as np
import matplotlib.pyplot as plt
import torch

from nets import *
from example_auto2D import *
from train_and_test import *



if __name__ == '__main__':

    autoencoder , cost_function , optimizer , ptrain = setup()

    xy0 , X = get_nn_input_gaussian( autoencoder.encoder.params , 2 )

    Jtrain = train( autoencoder , PytorchData4DTensor( X ) , optimizer , cost_function , ptrain )

    xylatent = autoencoder.get_latent_space_coordinates( X )
    
    plot_data( PytorchData4DTensor( X ) )
    plot_latent( xy0 )

    plot_latent( xylatent )

    plt.figure()
    plt.semilogy( Jtrain )

    plt.show()