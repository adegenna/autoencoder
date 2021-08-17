import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.backends.backend_pdf import PdfPages
import sys

from nets import *
from example_auto2D import *
from train_and_test import *


def multipage(filename, figs=None, dpi=200):
    
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()



if __name__ == '__main__':

    if torch.cuda.is_available():
        print( 'cuda available, using gpus' )

    print(f"output dir : {sys.argv[1]}")

    autoencoder , cost_function , optimizer , ptrain = setup( sys.argv[1] )

    xy0 , X = get_nn_input_gaussian( autoencoder.encoder.params , 2 )

    Jtrain = train( autoencoder , PytorchData4DTensor( X ) , optimizer , cost_function , ptrain )

    xylatent = autoencoder.get_latent_space_coordinates( X )
    
    plot_data( PytorchData4DTensor( X ) )
    plot_latent( xy0 )

    plot_latent( xylatent )

    plt.figure()
    plt.semilogy( Jtrain )

    multipage( sys.argv[1] + '/allfigs.pdf' , figs=None , dpi=200 )