import numpy as np
import matplotlib.pyplot as plt
import torch
import unittest

from nets import *


class Test_one( unittest.TestCase ):
    
    def setUp( self ):
        
        self.input_data_shape = Data4DTensorShape( 5,16,32,32 )
        self.encoder_params   = Conv2D_params( self.input_data_shape , \
                                               [ 8 , 4 ] , \
                                               [ 3 , 3 ] , \
                                                 2 )
        self.X = torch.tensor( np.ones( self.input_data_shape.get_pytorch_4dshape() ) ).float()
    
    def test_conv2d_hiddenLayerSize( self ):

        self.assertTrue( self.encoder_params.n_hidden_layers == 2 )

    def test_encoder( self ):

        encoder = Encoder_2D( self.encoder_params )
        self.assertTrue( encoder(self.X).shape == ( self.input_data_shape.batchsize , 1 , self.encoder_params.latent_space_dimn ) )

    def test_autoencoder( self ):

        autoencoder = AutoEncoder_2D( self.encoder_params )
        self.assertTrue( autoencoder(self.X).shape == self.X.shape )

if __name__ == '__main__':

    unittest.main()
