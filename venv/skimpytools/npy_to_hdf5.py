# -*- coding: utf-8 -*-
"""
.. module:: skimpy
   :platform: Unix, Windows
   :synopsis: Simple Kinetic Models in Python

.. moduleauthor:: SKiMPy team

[---------]

Copyright 2017 Laboratory of Computational Systems Biotechnology (LCSB),
Ecole Polytechnique Federale de Lausanne (EPFL), Switzerland

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIE CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""
from pytfa.io.json import load_json_model
from skimpy.io.yaml import load_yaml_model
from skimpy.analysis.oracle.load_pytfa_solution import load_fluxes, \
    load_concentrations, load_equilibrium_constants
from skimpy.sampling.simple_parameter_sampler import SimpleParameterSampler
from skimpy.core import *
from skimpy.mechanisms import *
from scipy.linalg import eigvals as eigenvalues
from sympy import Symbol
from skimpy.core.parameters import ParameterValues,ParameterValuePopulation
import pandas as pd
import numpy as np
import pickle
from skimpy.core.parameters import load_parameter_population

from sys import argv
#sys.path.append("..")

NCPU = 36
CONCENTRATION_SCALING = 1e6 # 1 mol to 1 mmol
TIME_SCALING = 1 # 1hr
# Parameters of the E. Coli cell
DENSITY = 1105 # g/L
GDW_GWW_RATIO = 0.38 # Assumes 70% Water

exp_id = 'fdp1'

path_to_kmodel  = '../models/kinetic/kin_varma_fdp1_curated.yml'
path_to_tmodel  = f'../models/thermo/varma_{exp_id}'
path_to_samples = f'../steady_state_samples/samples_{exp_id}.csv'

path_to_param_names_km = '../Models/parameter_names_km.pkl'
with open(path_to_param_names_km, 'rb') as input_file:
    parameter_names_km = pickle.load(input_file)

tmodel = load_json_model(path_to_tmodel)
kmodel = load_yaml_model(path_to_kmodel)
kmodel.prepare()
kmodel.compile_jacobian(sim_type=QSSA)
print('kmodel compiled')

# Load gan parameters and names as dataframe

#path_to_GAN_parameters = f'../gray_box_data/{exp_id}/sample_best.npy'
#path_to_max_eig = f'../gray_box_data/{exp_id}/sample_best_max_eig.csv'

# Load ORACLE parameters

path_to_ORACLE_parameters = f'../gan_input/{exp_id}/all_km_{exp_id}.npy'
path_to_max_eig =  f'../../../../skimpy/projects/kinvarma/small/kinetics/output/maximal_eigenvalues_{exp_id}.csv'

param = np.load(path_to_ORACLE_parameters)
eig = pd.read_csv(path_to_max_eig).iloc[:,1].values
idx_r = np.where(eig<=-9)[0]
idx_r = np.random.choice(idx_r, 1000)
param = param[idx_r,:]
#param = np.exp(param)       # GAN parameters are generated in log scale
parameter_set = pd.DataFrame(param)
parameter_set.columns  = parameter_names_km

##########################
# to load .hdf5 file though skimpy
'''
from skimpy.core.parameters import load_parameter_population
path_to_parameters = '/home/skimpy/work/Models_2/parameters_sample_id_0.hdf5'
param_pop = load_parameter_population(path_to_parameters)
'''
#Load ss fluxes and concentrations
samples = pd.read_csv(path_to_samples, header=0, index_col=0).iloc[0,0:]
flux_series = load_fluxes(samples, tmodel, kmodel,
                 density=DENSITY,
                 ratio_gdw_gww=GDW_GWW_RATIO,
                 concentration_scaling=CONCENTRATION_SCALING,
                 time_scaling=TIME_SCALING)

conc_series = load_concentrations(samples, tmodel, kmodel,
                                 concentration_scaling=CONCENTRATION_SCALING)
# Fetch equilibrium constants
k_eq = load_equilibrium_constants(samples, tmodel, kmodel,
                       concentration_scaling=CONCENTRATION_SCALING,
                       in_place=True)

symbolic_concentrations_dict = {Symbol(k):v for k,v in conc_series.items()}


sampling_parameters = SimpleParameterSampler.Parameters(n_samples=1)
sampler = SimpleParameterSampler(sampling_parameters)

sampler._compile_sampling_functions(kmodel, symbolic_concentrations_dict,  [])
model_param = kmodel.parameters
#idx_to_solve = np.random.randint(0,len(parameter_set.index),10)

stable_percent = 0
store_max_J = []

# to iterate over .hdf5 file sets
# for j in param_pop._index:

param_pop = []

for j in range(len(parameter_set.index)):

  if j%100==0:
        print(f'curr. set processed : {j}, stable % : {stable_percent*100/(j+0.001)}')

  param_val = parameter_set.loc[j]
  param_val = ParameterValues(param_val,kmodel)
  kmodel.parameters = k_eq
  kmodel.parameters = param_val
  parameter_sample = {v.symbol: v.value for k,v in kmodel.parameters.items()}
  #Set all vmax/flux parameters to 1.
  # TODO Generalize into Flux and Saturation parameters
  for this_reaction in kmodel.reactions.values():
    vmax_param = this_reaction.parameters.vmax_forward
    parameter_sample[vmax_param.symbol] = 1
    # Calculate the Vmax's

  kmodel.flux_parameter_function(
        kmodel,
        parameter_sample,
        symbolic_concentrations_dict,
        flux_series
        )
  for c in conc_series.index:
     if c in model_param:
        c_sym = kmodel.parameters[c].symbol
        parameter_sample[c_sym]=conc_series[c]

  this_param_sample = ParameterValues(parameter_sample,kmodel)
  param_pop.append(this_param_sample)

param_pop2 = ParameterValuePopulation(param_pop, kmodel = kmodel)
param_pop2.save(path_to_ORACLE_parameters.replace('npy', 'hdf5'))
print(path_to_ORACLE_parameters.replace('npy', 'hdf5'))


