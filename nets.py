import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from dataclasses import dataclass
from typing import List


@dataclass
class Data4DTensorShape:

    """
    input tensors to the autoencoder should be size = ( batchsize , channels , nX , nY )
    """

    batchsize : int
    channels  : int
    nX        : int
    nY        : int

    def get_pytorch_4dshape( self ):
        return ( self.batchsize , self.channels , self.nX , self.nY )

@dataclass
class PytorchData4DTensor:

    """
    really simple class for managing 4D tensors of shape ( batchsize , features , nX , nY )
    """

    X : torch.tensor
    
    def __post_init__( self ):

        self.batchsize , self.features , self.nX , self.nY = self.X.shape

    def get_data_elements( self , idx_data : int ):
        
        return self.X[ idx_data ]

    def get_features( self , idx_data : int , idx_feature : int ):

        return self.X[ idx_data , idx_feature ]
    


@dataclass
class Conv2D_params:

    dimn_tensor        : Data4DTensorShape
    hidden_layer_sizes : List[int]
    ksize              : List[int]
    latent_space_dimn  : int

    @property
    def n_hidden_layers( self ):
        return len( self.hidden_layer_sizes )

    @property
    def n_features_layers( self ):
        return [ self.dimn_tensor.channels ] + self.hidden_layer_sizes

    @property
    def n_features_layers_reversed( self ):
        return self.n_features_layers[::-1]

    @property
    def ksize_reversed( self ):
        return self.ksize[::-1]
    
    def get_moduleList( self ):

        return nn.ModuleList(
            [
                nn.Conv2d(
                    self.n_features_layers[i],
                    self.n_features_layers[i + 1],
                    kernel_size=self.ksize[i],
                    padding=(self.ksize[i] - 1) // 2,
                )
                for i in range(self.n_hidden_layers)
            ]
        )

    def get_moduleList_reversed( self ):

        return nn.ModuleList(
            [
                nn.Conv2d(
                    self.n_features_layers_reversed[i],
                    self.n_features_layers_reversed[i + 1],
                    kernel_size=self.ksize_reversed[i],
                    padding=(self.ksize_reversed[i] - 1) // 2,
                )
                for i in range( self.n_hidden_layers )
            ]
        )


class Encoder_2D(nn.Module):

    def __init__(self, params_c2d : Conv2D_params ):

        # input tensors are ( batchsize , channels , nX , nY )

        super().__init__()

        self.params = params_c2d

        # set up convolutional layers
        self.f_conv = self.params.get_moduleList()

        for conv_i in self.f_conv:
            nn.init.xavier_uniform_(conv_i.weight)

        # set up linear outout layer
        self.f_linear_out = nn.Linear(
            self.fc_outputsize,
            self.params.latent_space_dimn
        )

        nn.init.xavier_uniform_(self.f_linear_out.weight)
        
    @property
    def fc_outputsize( self ):

        """
        size of fc output right before latent space
        """
        
        return self.params.dimn_tensor.nX * self.params.dimn_tensor.nY * self.params.hidden_layer_sizes[-1]

    def forward(self, x):

        for conv_i in self.f_conv:
            x = F.relu(conv_i(x))
        
        x = self.f_linear_out( x.reshape( x.shape[0] , self.fc_outputsize) )

        return x


class Decoder_2D(nn.Module):

    def __init__( self, encoder : Encoder_2D ):

        # Input tensors are ( batchsize , latent_dimn )

        super().__init__()

        self.f_linear_in = nn.Linear( encoder.params.latent_space_dimn , encoder.fc_outputsize )

        nn.init.xavier_uniform_( self.f_linear_in.weight )

        self.f_conv = encoder.params.get_moduleList_reversed()

        for conv_i in self.f_conv:
            nn.init.xavier_uniform_(conv_i.weight)

        self.fc_outputsize = encoder.fc_outputsize

        self.params = Conv2D_params( encoder.params.dimn_tensor , 
                                     encoder.params.n_features_layers_reversed ,
                                     encoder.params.ksize_reversed ,
                                     encoder.params.latent_space_dimn )

    def forward(self, x):

        x = self.f_linear_in(x).reshape(
            x.size()[0], self.params.hidden_layer_sizes[0], self.params.dimn_tensor.nX, self.params.dimn_tensor.nY
        )

        for conv_i in self.f_conv[:-1]:
            x = conv_i(x)
            x = F.relu(x)

        x = self.f_conv[-1](x)

        return x


class AutoEncoder_2D(nn.Module):

    def __init__( self , params_c2d : Conv2D_params ):

        super().__init__()

        self.encoder = Encoder_2D( params_c2d )
        self.decoder = Decoder_2D( self.encoder )

    def forward(self, x):

        return self.decoder(self.encoder(x))

    def get_latent_space_coordinates(self, x):
        
        if torch.cuda.is_available():
            return np.squeeze( self.encoder(x.cuda()).detach().cpu().numpy() )
        return np.squeeze( self.encoder(x).detach().numpy() )
