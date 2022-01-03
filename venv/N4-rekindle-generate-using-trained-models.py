import os
import math
import h5py
import scipy
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import load_model
import helper as hp

n_samples = 1000      # No of models to generate
cond_class = -1       # -1 for relevant and 1 for non-relevant
noise = np.random.normal(0, 1, (n_samples, 127))

# Load generator
path_to_gen = 'path/to/generator.h5'
# Load scaling parameters
path_to_data_scaling = 'path/to/d_scaling.pkl'
with open(path_to_data_scaling, 'rb') as fp:
    d_scaling = pickle.load(fp)

sampled_labels = np.ones(n_samples).reshape(-1, 1) * cond_class
gen = load_model(path_to_gen)                       # load saved generator
gen_par = gen.predict([noise, sampled_labels])      # generate parameters

# rescale parameters
x_new, new_min, new_max = hp.unscale_range(gen_par, -1.0, 1.0, d_scaling['min_x'], d_scaling['max_x'])
# save parameters
os.makedirs('output', exist_ok = True)
np.save('output/generated_sample.npy', x_new)

