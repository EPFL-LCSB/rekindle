import pickle
from pytfa.io.json import load_json_model
from skimpy.io.yaml import load_yaml_model

'''
get metabolic subsytems of kinetic parameters
'''

exp_id = 'fdp1'

path_to_param_names_km = 'Models/parameter_names_km.pkl'
with open(path_to_param_names_km, 'rb') as input_file:
     parameter_names_km = pickle.load(input_file)

path_to_kmodel  = '../models/kinetic/kin_varma_fdp1_curated.yml'
path_to_tmodel  = f'../models/thermo/varma_{exp_id}'

tmodel = load_json_model(path_to_tmodel)
kmodel = load_yaml_model(path_to_kmodel)

km_subsystems= []

for km in parameter_names_km:
    rxn_name = km.split('_', 2)[-1]
    km_subsytems.append(getattr(tmodel.reactions, rxn_name).subsystem)

pd.to_csv(km_subsytem, path_to_model.replace(f'varma_{exp_id}', 'km_subsytems.csv'))
