from pytfa.io.json import load_json_model
from skimpy.io.yaml import load_yaml_model
from skimpy.analysis.oracle.load_pytfa_solution import load_fluxes, \
    load_concentrations, load_equilibrium_constants
from skimpy.analysis.ode import sample_initial_concentrations
from skimpy.core.solution import ODESolutionPopulation
from skimpy.core.parameters import ParameterValuePopulation, \
    load_parameter_population
from skimpy.utils.namespace import QSSA
from skimpy.utils.tabdict import TabDict


import os
import sys
import numpy as np
import pandas as pd

sys.path.append("..")

# Parameters
TOUT = np.logspace(-9, 1, 1000) # 1hr
CONCENTRATION_SCALING = 1e6 # 1 mol to 1 mmol

exp_id = 'fdp1'  #<--- Select 1 out of 4 Physiologies (fdp1-fdp4)

# Paths
path_to_kmodel = '../models/kinetic/kin_varma_curated.yml'
path_to_tmodel = f'../models/thermo/varma_{exp_id}'
path_to_samples = f'../models/steady_state_samples/samples_{exp_id}.csv'

path_to_output = f'ode_output/' #<--- Path to output
os.makedirs(path_to_output, exist_ok=True)

#load kinetic parameter_sets.hdf5
path_to_params = '/path/to/parameter_sets.hdf5'  # must be in .hdf5

# load models
tmodel = load_json_model(path_to_tmodel)
kmodel = load_yaml_model(path_to_kmodel)
parameter_population = load_parameter_population(path_to_params)

# Load steady state TFA samples
samples = pd.read_csv(path_to_samples, header=0, index_col=0)

# Compile ODE-Functions
kmodel.prepare()
kmodel.compile_ode(sim_type=QSSA,ncpu=32)

solutions = []
this_samples = list(parameter_population._index.keys())
samples_to_simulate = this_samples

n_perturbations = 10  # Number of unique random perturbations to each parameterized model

for R in range(n_perturbations):

     path_to_output = f"ode_output//{exp_id}/"
     path_to_output = path_to_output.replace(f'{exp_id}/', f'{exp_id}_{R}/')
     os.makedirs(path_to_output,exist_ok=True)

     for ix in samples_to_simulate:
        print(ix)
        if os.path.isfile(path_to_output.format(ix)):
            continue
        # Get reference concentrations

        #tfa_id,_ = ix.split(',')
        #tfa_id = int(tfa_id)
        sample = samples.loc[0]
        reference_concentrations = load_concentrations(sample, tmodel, kmodel,
                                             concentration_scaling=CONCENTRATION_SCALING)

        #Add parameters
        kmodel.parameters = parameter_population[ix]

        # Sample perturbations with consitent perturbations
        perturbed_concentrations = sample_initial_concentrations(kmodel,
                                      reference_concentrations,
                                      lower_bound=0.5,   # mininum perturbation allowed to the steady state
                                      upper_bound=2,     # maximum perturbation allowed to the steady state
                                      n_samples=1)       # no of steady states to be evaluated

        # Function to stop integration
        def rootfn(t, y, g, user_data):
            y0 = user_data['y0']
            n_max = user_data['max_norm']
            norm = np.sqrt(np.square((y-y0)/y0).sum())
            #norm = np.sqrt(np.square((y-y0)).sum())
            if norm >= n_max:
                g[0] = 0
            else:
                g[0] = 1

        user_data = {'y0':reference_concentrations[kmodel.variables].values,
                     'max_norm':1e5}
        solutions = []
        # TODO make this a parallel loop
        for i, concentrations in perturbed_concentrations.iterrows():
            # Integrate perturbations
            kmodel.initial_conditions = TabDict([(k,v) for k,v in concentrations.iteritems()])
            this_sol_qssa = kmodel.solve_ode(TOUT,
                                             solver_type='cvode',
                                             rtol=1e-9,
                                             atol=1e-9,
                                             max_steps=1e9,
                                             rootfn=rootfn,
                                             nr_rootfns=1,
                                             user_data=user_data
                                             )
            # Normalize the concentrations
            variables = this_sol_qssa.concentrations.columns
            this_sol_qssa.concentrations = (this_sol_qssa.concentrations - reference_concentrations[variables])\
                                       /reference_concentrations[variables]
            #this_sol_qssa.concentrations = this_sol_qssa.concentrations #/ reference_concentrations[variables]

            solutions.append(this_sol_qssa)
            print('Compute solution {}'.format(i))

        #save_name = this_id.replace('.hdf5','')
        save_name = f'solution_fdp{id_}'
        solpop = ODESolutionPopulation(solutions)

        #save time series ODe integrated solution as .csv file
        solpop.data.to_csv(f'{path_to_output}{save_name}_{ix}.csv')

        #save the reference steady state concentrations
        reference_concentrations[variables].to_csv(f'{path_to_output}/reference_concentrations.csv')

        #Clear Memory
        del solpop
