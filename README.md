![here](https://github.com/EPFL-LCSB/rekindle/blob/master/venv/cGANtools/rekindle-logo_2.png)

ReKinDLe (Reconstruction of Kinetic models using Deep Learning) is a python package for training and generating with generative adversarial networks (GANs) to parametrize large-scale nonlinear kinetic models of cellular metabolism

## Installation
First, you need to clone the repository to your local machine

    git clone https://github.com/EPFL-LCSB/rekindle.git /path/to/rekindle
    cd /path/to/rekindle

Then, you create a conda virtual environement and install all the needed packages for GAN training:
    
    conda env create -f rekindle_env.yml

### Additional requirements
Additionally, you will need to install SkimPy to work run certain scripts. Check the instructions to install SkimPy [here](https://github.com/EPFL-LCSB/skimpy/).<b> Note: SkimPy is not a part of the 
rekindle environment, you have to install it separately. </b>

The models used in this module are publicly available [here](https://zenodo.org/record/5803120#).

## How to use

First, change directory to the virtual envvironment folder.

    cd venv

#### 1. Create training data 

We first preprocess our data (kinetic parameter sets from ORACLE) to make it ready to be fed to the GANs. This raw data is present in models/parameters/
To begin preprocessing run the following command.

    python N1-rekindle-data-preprocessing.py
    
Alternatively, you can also run the script in IPython and view the objects and variables by running the following commands,
 
    ipython
    run N1-rekindle-data-preprocessing.py
 
The pre-processed raw data should now be present in gan_input/fdp1/
#### 2. Train the GANs
To begin training run the following script in the command line or in IPython,
 
    python N2-rekindle-do-cgan-training.py
    
The training metrics (losses, accuracy and their plots), saved generator models (.h5) and GAN generated parameter samples will be stored in gan_output/fdp1/repeat_0/
Change the repeat hyperparameter in the script to have multiple repeats.
  
#### 3. Calculate the maximal eigenvalues of the Jacobian of the generated samples
  
<b>NOTE</b>: SkimPy is a requirement for this step.
To calculate the eigenvalues run the following script,
  
    python N3-skimpy-calculate-eigenvalues.py
  
The maximal eigenvalues will be stored in the same folder as the previous step.
  
#### 4. Generate using pre-trained models 
  
To load a pre-trained generator add the path to the generator in the script and then to generate from it run the following command,
      
    python N4-rekindle-generate-using-trained-models.py
    
The generated sample will be stored in output/. 

##### 5. Integrate parameterized ordinary differential equations (ODEs)
<b>NOTE:</b> SkimPy is a requirement for this step.

Additionaly,to integrate the parameterized ODEs follow the next two steps. To convert a parameter sample from .npy to .hdf5 run the following command,

    python N5-skimpy-covert-parameters-to-hdf5.py
    
To integrate the parameterized ODEs run the following script after setting the parameter path variable to the .hdf5 file generated in the previous step,

    python N6-skimpy-integrate-ODEs.py
      
The integrated time-series solutions will be stored in the folder ode_output/

The kinetic nonlinear models are available [here](https://zenodo.org/record/5803120#).  
   
## License

The software in this repository is put under an APACHE-2.0 licensing scheme - please see the [LICENSE](https://github.com/EPFL-LCSB/rekindle/blob/master/LICENSE) file for more details.
 
 
