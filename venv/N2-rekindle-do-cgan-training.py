import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from cGANtools.GAN import CGAN

if __name__ == "__main__":

    # Training hyperparameters

    latent_dim = 127      # Length of noise vector
    epochs = 10           # Total number of epochs
    n_sample = 300        # No of parameter sets to generate at every sampling epoch
    repeats = 2           # number of training repeats
    batchsize = 1024      # Batchsize
    sample_interval = 10  # Frequency of testing generator

    exp_id = 'fdp1'       #<-- Choose 1 of 4 physiologies (fdp1-fdp4)
    path_generator = None #<---if doing transfer learning put path to trained generator here else leave None
                          #    if loading model using load_model gives an error upgrade tensorflow to v2.3.0
                          #    > pip install tensorflow==2.3.0

    print('\nSTARTING CGAN TRAINING')

    # load the data for appropriate experiment
    datapath = f'gan_input/{exp_id}/'
    X_train = np.load(f'{datapath}X_train_{exp_id}.npy')
    y_train = np.load(f'{datapath}y_train_{exp_id}.npy')

    # Specify output folders
    savepath = f'gan_output/{exp_id}/'

    for j in range(0, repeats):

        print(f'Current exp: {exp_id}: Samples used: {np.shape(X_train)[0]}, repeat {j}')
        # set save directory
        this_savepath = f'{savepath}repeat_{j}/'
        os.makedirs(this_savepath, exist_ok=True)

        cgan = CGAN(X_train, y_train, latent_dim, batchsize, path_generator, savepath= this_savepath)
        d_loss, g_loss, acc = cgan.train(epochs, sample_interval, n_sample)

        # store training summary

        this_train_savepath = f'{this_savepath}training_summary/'
        os.makedirs(this_train_savepath, exist_ok=True)

        with open(f'{this_train_savepath}d_loss.pkl', 'wb') as f:
            pickle.dump(d_loss, f)
        with open(f'{this_train_savepath}g_loss.pkl', 'wb') as f:
            pickle.dump(g_loss, f)
        with open(f'{this_train_savepath}acc.pkl', 'wb') as f:
            pickle.dump(acc, f)

        # plot metrics
        x_plot = np.arange(0, epochs, 1)

        fig = plt.figure(figsize = (10,5))
        ax1 = fig.add_axes([0.2, 0.2, 1, 1])

        ax1.plot(x_plot, d_loss, label = 'discriminator loss')
        ax1.plot(x_plot, g_loss, label= 'generator loss')
        ax1.set( ylabel = 'criterion_losses', xlabel = 'epochs')
        ax1.legend()
        plt.savefig(f'{this_train_savepath}loss.svg', dpi=300,
                    transparent=False, bbox_inches='tight')

        fig = plt.figure(figsize= (10,5))
        ax2 = fig.add_axes([0.2, 0.2, 1, 1])

        ax2.plot(x_plot, d_loss, label='discriminator accuracy')
        ax2.set(ylabel='accuracy', xlabel='epochs')
        ax2.legend()
        plt.savefig(f'{this_train_savepath}d_accuracy.svg', dpi=300,
                    transparent=False, bbox_inches='tight')

