import os
import pickle
import numpy as np
import pandas as pd

#import eigenvalue calculator
from skimpytools.stability import CheckStability

exp_id = 'fdp1' #<-- Choose 1 of 4 physiologies (fdp1-fdp4)
# Load parameter set (must be .npy)
path_to_GAN_generated_parameters = f'gan_output/{exp_id}/repeat_0/0_r.npy'
GAN_generated_parameters = np.load(path_to_GAN_generated_parameters)

# Load parameter names
path_to_param_names_km = f'gan_input/{exp_id}/parameter_names_km_{exp_id}.pkl'
with open(path_to_param_names_km, 'rb') as input_file:
     parameter_names_km = pickle.load(input_file)

checkstability = CheckStability()

# Load kinetic model, thermodynamic model and reference steady state for a given physiology
checkstability._load_models(exp_id)

# Calculate eigenvalues
'''
Loop over the next block of code if using multiple times (not the _load_model above)
'''
checkstability._prepare_models(GAN_generated_parameters, parameter_names_km)
_,max_eig = checkstability.calc_eigenvalues_recal_vmax()

#store and save the maximal eigenvalues
pd.DataFrame(max_eig).to_csv(path_to_GAN_generated_parameters.replace('.npy','eigenvalues.csv'))

