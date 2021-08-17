import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import List , Callable , Tuple
from dataclasses import dataclass

from nets import *


@dataclass
class TrainingDataParameters:

    epochs            : int
    checkpoint_period : int
    minibatch_size    : int
    fileout_name      : str = './nn_'


def train( autoencoder   : AutoEncoder_2D , 
           X             : PytorchData4DTensor ,
           optimizer     : torch.optim ,
           cost_function : Callable[ [ torch.tensor , torch.tensor ] , float ] , 
           ptrain        : TrainingDataParameters ):

    def cuda_cost( xj ):
        if torch.cuda.is_available():
            return cost_function( Variable( xj ).cuda() , autoencoder( Variable( xj ).cuda() ) )
        return cost_function( Variable( xj ) , autoencoder( Variable( xj ) ) )

    batchsize , features , nX , nY = autoencoder.encoder.params.dimn_tensor.get_pytorch_4dshape()

    J = []
    
    for i in range( ptrain.epochs ):

        xi   = X.get_data_elements( torch.randint( 0 , batchsize , ( ptrain.minibatch_size , ) ) )
        
        optimizer.zero_grad()
        cost = 0

        for j in range( batchsize // ptrain.minibatch_size ):

            xj   = X.get_data_elements( torch.randint( 0 , batchsize , ( ptrain.minibatch_size , ) ) )
            
            cost  = cuda_cost( xj )

            cost.backward()
        
        optimizer.step()
        J.append( cost )
        
        print("EPOCH = %d, COST = %.6f" %(i+1,cost))

        if ( (i+1) % ptrain.checkpoint_period == 0 ):
            print( "Saving current autoencoder to disk" )
            torch.save( autoencoder , ptrain.fileout_name + str(i+1) + '.pt' )
            
    return J



def test( net           : AutoEncoder_2D , 
          x_test        : PytorchData4DTensor , 
          y_test        : PytorchData4DTensor , 
          cost_function : Callable[ [ torch.tensor , torch.tensor ] , float ] , 
          savefile      : str = None ) -> Tuple[ PytorchData4DTensor , float ]:
    
    if ( savefile is not None ):
        net = torch.load( savefile )
        net.eval()
    
    y_pred     = PytorchData4DTensor( net( Variable( x_test.X ) ) )
    y_pred     = y_pred.X.detach().numpy()
    y_test     = y_test.X.detach().numpy()

    J = np.sum( [ cost_function( pi , yi ) for ( pi , yi ) in zip( y_pred.X , y_test.X ) ] )

    return y_pred , J