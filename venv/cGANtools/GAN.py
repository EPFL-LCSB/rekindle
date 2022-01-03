import os, sys
import math
import time
import pickle
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from random import sample
from functools import partial

import tensorflow.keras.backend as K

from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.models import Sequential, Model, load_model
from keras.optimizers.schedules import ExponentialDecay
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, Dense, Dropout, concatenate, BatchNormalization

import helper as hp


#check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


class CGAN():
    def __init__(self, X_train, y_train, latent_dim, batch_size, path_generator, savepath, num_classes=2, verbose = False):

        self.param_shape = X_train.shape[1]
        self.label_shape = 1
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.batchsize = batch_size
        self.savepath = savepath
        self.verbose = verbose
        self.min_x = None
        self.max_x = None
        self.informed_labelling = False

        if path_generator == None:
            self.transfer_learning = False
        else:
            self.transfer_learning = True
            self.path_generator = path_generator

        # data
        self.X_train = X_train
        self.y_train = y_train

        # Build Optimizer
        samples_per_epoch = int(X_train.shape[0] / self.batchsize)

        # Build Learning rate scheduler
        initial_lr = 0.001
        # lr_schedule=ExponentialDecay(initial_lr,decay_steps = samples_per_epoch*50, decay_rate = 0.90, staircase = True)
        # lrate = LearningRateScheduler(self.step_decay)
        # optimizer = Adam(learning_rate=initial_lr, beta_1=0.5)
        optimizer = Adam(0.0002, 0.5)
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        d_trainable_count = np.sum([K.count_params(w) for w in self.discriminator.trainable_weights])
        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss=['binary_crossentropy'],
                               optimizer=optimizer, metrics='accuracy')

        g_trainable_count = np.sum([K.count_params(w) for w in self.generator.trainable_weights])
        g_non_trainable_count = np.sum([K.count_params(w) for w in self.generator.non_trainable_weights])

        print(f'Total trainable parameters: {g_trainable_count + d_trainable_count}')

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.label_shape,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated parameteres as input and determines validity
        # and the label of that parameters
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates parameters => determines validity
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
                              optimizer=optimizer, metrics='accuracy')

    def build_generator(self):

        if self.transfer_learning == False:

            model = Sequential()
            model.add(Dense(128, input_dim=self.latent_dim + self.label_shape))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dropout(0.5))
            model.add(Dense(256))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dropout(0.5))
            model.add(Dense(512))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dropout(0.5))
            model.add(Dense(self.param_shape, activation='tanh'))

            if self.verbose:
                model.summary()

            noise = Input(shape=(self.latent_dim,))
            label = Input(shape=(self.label_shape,))

            model_input = concatenate([noise, label])

            param = model(model_input)

            return Model([noise, label], param)

        else:
            model = load_model(self.path_generator)
            return model

    def build_discriminator(self):

        model = Sequential()
        model.add(Dense(32, input_dim=self.param_shape + self.label_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))

        if self.verbose:
            model.summary()

        param = Input(shape=(self.param_shape,))
        label = Input(shape=(self.label_shape,))

        model_input = concatenate([param, label])

        validity = model(model_input)

        return Model([param, label], validity)

    def train(self, epochs, sample_interval, n_samples):

        # Rescale the input between -1 to 1
        # This is needed because we use the Tanh as output
        # function, therefore we need to match the domain
        # of definitino of that function
        X_train, min_x, max_x = hp.scale_range(self.X_train, -1.0, 1.0)
        self.min_x = min_x
        self.max_x = max_x

        # save for future sampling
        d_scaling = {'min_x': min_x,
                     'max_x': max_x}
        hp.save_pkl(f'{self.savepath}d_scaling.pkl', d_scaling)

        batchsize = self.batchsize
        half_batch = int(batchsize / 2)

        all_d_loss = []
        all_g_loss = []
        all_acc = []

        # Compute how many samples will
        # go in a batch
        samples_per_epoch = X_train.shape[0]
        number_of_batches = int(samples_per_epoch / batchsize)
        # epoch_tsr = tf.Variable(0, trainable=False, name='Epochs', dtype=tf.int64)

        for epoch in range(epochs):

            # print(f'Number of batches: {number_of_batches}')
            epoch_g_loss = []
            epoch_d_loss = []
            epoch_acc = []

            # decaying_learning_rate = partial(decay_func, epoch_tensor=epoch_tsr)

            for i in range(number_of_batches):
                X_batch = np.array(X_train[batchsize * i:batchsize * (i + 1)])
                y_batch = np.array(self.y_train[batchsize * i:batchsize * (i + 1)])

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random half batch

                idx = np.random.randint(0, X_batch.shape[0], half_batch)
                params, labels = X_batch[idx], y_batch[idx]

                noise = np.random.normal(0, 1, (half_batch, self.latent_dim))

                # Generate a half batch
                gen_params = self.generator.predict([noise, labels])

                valid = np.ones((half_batch, 1))  # +np.random.normal(0,0.8,(half_batch,1))
                fake = np.zeros((half_batch, 1))  # +abs(np.random.normal(0,0.5,(half_batch,1)))

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch([params, labels], valid)
                d_loss_fake = self.discriminator.train_on_batch([gen_params, labels], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                noise = np.random.normal(0, 1, (batchsize, self.latent_dim))

                valid = np.ones((batchsize, 1))  # +np.random.normal(0,0.8,(batch_size,1))

                # Generator wants discriminator to label the generated images as the intended
                # stability

                data_labels = [1, -1]
                sampled_labels = np.random.choice(data_labels, batchsize)

                # sampled_labels = np.random.randint(0,2,batch_size)

                # Add noise
                sampled_labels = np.array([j + np.random.normal(0, 0) for j in sampled_labels])
                sampled_labels = sampled_labels.reshape((-1, 1))

                # Train the generator
                g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

                epoch_d_loss.append(d_loss[0])
                epoch_g_loss.append(g_loss)
                epoch_acc.append(100 * d_loss[1])

            all_d_loss.append(np.mean(epoch_d_loss))
            all_g_loss.append(np.mean(epoch_g_loss))
            all_acc.append(np.mean(epoch_acc))

            # Discriminator overpower check
            if epoch >= 200:
                moving_average = np.mean(all_acc[-200:])
                if moving_average >= 90:
                    print(f'Moving average: {moving_average}')
                    break

            # Plot the progress
            mean_d_loss = np.mean(epoch_d_loss)
            mean_acc = np.mean(epoch_acc)
            mean_g_loss = np.mean(epoch_g_loss)


            print(f'Epoch {epoch}, D loss: {mean_d_loss}, acc: {mean_acc}, G loss: {mean_g_loss}')

            # Generate data at every sample interval

            if epoch % sample_interval == 0:
                if self.num_classes == 2:
                    self.sample_parameters(epoch, n_samples, cond_class=-1)
                    # self.sample_parameters(epoch, n_samples, cond_class=1)
                else:
                    raise ValueError('The current code works with two classes for now')

        return all_d_loss, all_g_loss, all_acc

    def sample_parameters(self, epoch, n_samples, cond_class):

        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))

        # Create the conditional label for cond_class
        sampled_labels = np.ones(n_samples).reshape(-1, 1) * cond_class
        gen_par = self.generator.predict([noise, sampled_labels])

        # Rescale parameters according to previous scaling on X_train
        x_new, new_min, new_max = hp.unscale_range(gen_par, -1.0, 1.0, self.min_x, self.max_x)
        class_label = 'r' if cond_class==-1 else 'nr'
        np.save(f'{self.savepath}{epoch}_{class_label}.npy', x_new)

        # and save the corresponding generator and descriminator
        path_models = f'{self.savepath}saved_models/'
        os.makedirs(path_models, exist_ok=True)

        self.generator.save(f'{path_models}generator_{epoch}.h5')
        #    self.discriminator.save(f'{path_models}discriminator_{epoch}.h5')


