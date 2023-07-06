import numpy as numpy
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from typing import List , Callable, Tuple , Union
from train_and_test import TrainingDataParameters

from nets import *


def setup( output_dir : str ):

    input_data_shape = Data4DTensorShape( 128,1,128,128 )
    encoder_params   = Conv2D_params( input_data_shape , \
                                    [ 4 , 8 , 16 , 32 ] , \
                                    [ 3 , 3 , 3  , 3 ] , \
                                    2 )

    net           = AutoEncoder_2D( encoder_params )
    cost_function = torch.nn.MSELoss()
    optimizer     = torch.optim.Adam( net.parameters() , lr=0.001 )
    ptrain        = TrainingDataParameters( 1000 , 100 , 16 , output_dir + '/nn_' )

    if torch.cuda.is_available():
        net = net.cuda()

    return net , cost_function , optimizer , ptrain


def make_gaussian_data( x0 , y0 , sigma , params : Conv2D_params ):

    xx,yy = np.meshgrid( np.arange( params.dimn_tensor.nX ) , np.arange( params.dimn_tensor.nY ) )

    return np.exp( -0.5 * ( (xx-x0)**2 + (yy-y0)**2 )**2 / sigma**2 )


def get_nn_input_gaussian( params : Conv2D_params , d_compressed : int = 2 ) -> torch.tensor:
    
    """
    d_compressed : true underlying dimension 
    n_samples : number of mc samples
    """

    x0 = params.dimn_tensor.nX//4 * np.cos( 2*np.pi*np.linspace( 0 , 1 , params.dimn_tensor.batchsize ) ) + params.dimn_tensor.nX//2
    y0 = params.dimn_tensor.nY//4 * np.sin( 2*np.pi*np.linspace( 0 , 1 , params.dimn_tensor.batchsize ) ) + params.dimn_tensor.nY//2

    X = np.expand_dims( np.array( [ make_gaussian_data( \
        x0i , y0i , params.dimn_tensor.nX , params ) \
            for (x0i,y0i) in zip(x0,y0) ] ) , 1 ) # (batchsize , 1 , nX , nY)

    X = torch.tensor( np.reshape( X , params.dimn_tensor.get_pytorch_4dshape() ) ).float()

    return list(zip(x0,y0)) , X


def plot_latent( xy_2d : List[ Tuple[ float , float ] ] ):
    
    plt.figure()
    plt.scatter( [ xi for xi,yi in xy_2d ] , [ yi for xi,yi in xy_2d ] , c=np.arange(len(xy_2d)) , cmap=plt.cm.get_cmap('coolwarm') )
    plt.gca().set_aspect('equal')


def plot_data( X : PytorchData4DTensor , plotx : int = 2 , ploty : int = 2 ):

    xx,yy = np.meshgrid( np.arange( X.nX ) , np.arange( X.nY ) )
    
    plt.figure()
    for i in range( plotx * ploty ):
        plt.subplot( plotx , ploty , i+1 )
        plt.contourf( xx , yy , np.squeeze( X.get_data_elements( i * X.batchsize // (plotx*ploty) ) ) )
        plt.gca().set_aspect('equal')


def load_and_analyze_results( savefile : str , 
                              X : PytorchData4DTensor = None,
                              xy0 : List[ Tuple[ float , float ] ] = None, 
                              plotx : int = 2 ,
                              ploty : int = 2 ):

    net = torch.load( savefile )

    if X is None:
        xy0 , X = get_nn_input_gaussian( net.encoder.params , 2 )
        X = PytorchData4DTensor(X)

    if xy0 is not None:
        plot_latent( xy0 )

    data = X.X.to('cuda')
    net = net.to('cuda')

    plot_data( X , plotx , ploty )
    plot_data( PytorchData4DTensor( net( Variable(data) ).cpu().detach().numpy() ) , plotx , ploty )
    plot_latent( net.get_latent_space_coordinates( data ) )


if __name__ == '__main__':

    load_and_analyze_results( input('pt savefile : ') )

    plt.show()