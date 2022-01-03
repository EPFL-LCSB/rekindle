
## Imports
import os, sys
import time
import yaml
import h5py
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    start = time.time()
    print('\nSTART PROCESSING')

    # pre-processing parameters
    exp_id = 'fdp1'  #<--- Select 1 out of 4 Physiologies (fdp1-fdp4)
    parameter_set_dim = 411  # No of kinetic parameters in model

    # fetch training set parameters
    path_parameters = f'models/parameters/parameters_sample_id_{exp_id}_0.hdf5'
    if not path_parameters.endswith('.hdf5'):
        raise ValueError('Your data must be a .hdf5 file')

    # fetch training set eigenvalues
    path_stability = f'models/parameters/maximal_eigenvalues_{exp_id}.csv'
    if not path_stability.endswith('.csv'):
        raise ValueError('Your data must be a .csv file')


    # get the data and processed
    f = h5py.File(path_parameters, 'r')
    stabilities = pd.read_csv(path_stability).iloc[:, 1].values


    n_parameters = f[('num_parameters_sets')][()]
    all_data = np.empty([n_parameters, parameter_set_dim])
    all_stabilities = np.empty([n_parameters])

    J_partition = -9  #<--- Create class partition based on this eigenvalue
    count0, count1 = 0, 0

    for i in range(0, n_parameters):

        if i % 10000 == 0:
            print(f'current set processed: {i}')
        this_param_set = f'parameter_set_{i}'
        param_values = np.array(f.get(this_param_set))

        mreal = stabilities[i]

        if mreal >= J_partition:
            stability = 1
            count0 += 1
        elif mreal < J_partition:
            stability = -1
            count1 += 1

        all_data[i] = param_values
        all_stabilities[i] = stability

    all_data = np.array(all_data)
    all_stabilities = np.array(all_stabilities)

    n_param = all_data.shape[0]
    print(f'% relevant models: {count1 / n_param}')

    # keep only km
    parameter_names = list(f['parameter_names'])
    idx_to_keep = [i for i, x in enumerate(parameter_names) if 'km_' in x]
    all_km = all_data[:, idx_to_keep]
    all_km_names = [x for i, x in enumerate(parameter_names) if 'km_' in x]

    print(f'Shape of all data: {all_km.shape}')

    # take the log
    log_all_data = np.log(all_km)  #Log transform all parameters

    # train-val split
    ratio = float(0.9)  # Partition of training and test data
    n_data = log_all_data.shape[0]
    limit = int(ratio * n_data)
    all_idx = np.arange(n_data)
    np.random.shuffle(all_idx)

    idx_tr = all_idx[:limit]
    idx_val = all_idx[limit:]

    tr_data = log_all_data[idx_tr]
    val_data = log_all_data[idx_val]

    tr_stabi = all_stabilities[idx_tr]
    val_stabi = all_stabilities[idx_val]

    print(f'N data for training: {tr_data.shape[0]}')
    print(f'N data for validation: {val_data.shape[0]}')

    # save everything
    savepath = f'gan_input/{exp_id}'
    os.makedirs(savepath, exist_ok=True)
    np.save(f'{savepath}/all_km_{exp_id}.npy', all_km)
    np.save(f'{savepath}/all_targets_{exp_id}.npy', all_stabilities)
    np.save(f'{savepath}/X_train_{exp_id}.npy', tr_data)
    np.save(f'{savepath}/X_val_{exp_id}.npy', val_data)

    np.save(f'{savepath}/y_train_{exp_id}.npy', tr_stabi)
    np.save(f'{savepath}/y_val_{exp_id}.npy', val_stabi)

    with open(f'{savepath}/parameter_names_{exp_id}.pkl', 'wb') as f:
        pickle.dump(parameter_names, f)
    with open(f'{savepath}/parameter_names_km_{exp_id}.pkl', 'wb') as f:
        pickle.dump(all_km_names, f)

    end = time.time()
    print(f'PROCESSING DONE in {end - start:.05} seconds')