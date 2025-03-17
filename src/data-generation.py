


# Libraries
import os
import time
from tqdm import tqdm
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyG
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "torch_geometric.typing")
warnings.filterwarnings("ignore", category=UserWarning, module = "torch_geometric.deprecation")

from torch_geometric.data import Data, DataLoader


def deviceSetup():
    '''
    Function to setup the device for training

    Args:
        -

    Returns:
        device (torch.device) = Device to be used for training
        device_ids (list)     = List of device ids
        n_gpus (int)          = Number of GPUs available
    '''

    # CPU as default device
    device_ids = []
    n_gpus = 0
    device = torch.device("cpu")

    # Check if GPU is available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        n_gpus = torch.cuda.device_count()
        device_ids = list(range(n_gpus))
        device = torch.device(f"cuda:{device_ids[0]}")

    return device, device_ids, n_gpus


def generateNBodyData(category, config):
    '''
    Function to generate N-Body simulation based on Section 5.1 of the paper

    Args:
        category (str) = Category of the dataset (train, val, test)
        config (dict)  = Dict with required parameters 

    Returns:
        gen_sim (list) = List of generated simulations
    '''

    # Unpack values
    n_particles = config['n_particles']                 # Num of particles
    n_steps     = config['n_steps']                     # Num of timesteps
    dt          = config['dt']                          # Time step size used in integration
    noise       = config['noise']                       # s.d. of Gaussian noise


    # Based on the category, find the number of independent simulations
    if category == "train":
        n_sims = config['n_train']
    elif category == "val":
        n_sims = config['n_val']
    elif category == "test":
        n_sims = config['n_test']
    else:
        raise ValueError("Invalid category. Choose from 'train', 'val', 'test'")


    # Init list to store simulations
    gen_sim = []

    # Loop through the number of simulations
    for i in tqdm(range(n_sims), desc = "Running simulations for dataset generation"):

        # Initialize array to store trajectories
        # Shape: [timesteps x num of particles x 3D coordinates]
        pos = np.zeros(
            (n_steps, n_particles, 3)
        )

        vel = np.zeros(
            (n_steps, n_particles, 3)
        )

        # Initial position and velocity in a *bounded region*
        # Shape: [num of particles x 3D coordinates]
        pos[0] = np.random.uniform( -1, 1, (n_particles, 3) )
        vel[0] = np.random.uniform( -0.5, 0.5, (n_particles, 3) )

        # Extract the current position and velocity
        curr_pos, curr_vel = pos[0], vel[0]

        # Random charges
        # Shape: [num of particles]
        charges = np.random.choice([-1.0, 1.0], size = n_particles)

        # Loop through the number of timesteps
        for t in range(1, n_steps):

            # List to store force between particles
            # Shape: [num of particles x 3D coordinates]
            force = np.zeros( (n_particles, 3) )

            # Loop through the number of particles
            # Find the force between each pair of particles (except with itself)
            for i in range(n_particles):
                for j in range(n_particles):

                    if i != j:

                        # Calculate the distance between particles
                        d_indiv = curr_pos[i] - curr_pos[j]
                        dist = np.linalg.norm( d_indiv )

                        # if dist is too small, set it to 0.1
                        dist = max(dist, 0.1)

                        # Calculate the force between particles using Coulomb's Law
                        # F = k * q1 * q2 / r^2
                        f = charges[i] * charges[j] / dist**2
                        force[i] += f * d_indiv


            # Update the velocity and position of particles
            # v = v + F * dt and x = x + v * dt
            # Smaller dt => more accurate simulation but slower
            curr_vel += force * dt
            curr_pos += curr_vel * dt

            # Store the updated position and velocity
            pos[t] = curr_pos.copy()
            vel[t] = curr_vel.copy()

        # Add noise to the positions
        pos += np.random.normal(0, noise, pos.shape)

        # Store the simulation
        gen_sim.append( (pos, vel, charges) )
    
    return gen_sim



def convertDatasetToPyG(generated_sim):
    '''
    Function to convert the raw generated simulation to PyTorch Geometric Data object.

    Args:
        generated_sim (list) = List of generated simulations

    Returns:
        py_data (list) = List of PyTorch Geometric Data objects
    '''

    # Init list to store PyG data
    py_data = []

    # Loop through the generated simulations
    for p, v, c in generated_sim:

        # Extract num_time_steps and num_particles from the simulation
        n_ts, n_particles, _ = p.shape

        # INPUT: Initial position and velocity as tensor
        x_0 = torch.tensor(p[0], dtype = torch.float)
        v_0 = torch.tensor(v[0], dtype = torch.float)

        # compute velocity norms: v_norm = sqrt(v_x^2 + v_y^2 + v_z^2)
        # USED as E(n) invariant feature as mentioned in Section 3.1
        v_norm = torch.norm(v_0, dim = 1, keepdim = True)

        # TARGET: Final position as tensor
        x_n = torch.tensor(p[-1], dtype = torch.float)

        # Charges --> E(n) invariant scalar feature
        c_inv = torch.tensor(c, dtype = torch.float)

        # Data object
        data = Data(
            x = x_0,
            v = v_0,
            v_norm = v_norm,
            c = c_inv,
            y = x_n
        )
        py_data.append(data)

    return py_data



def cacheDataset(data_gen_config):
    '''
    Function to cache dataset generated for faster execution

    Args:
        data_gen_config (dict) = Dictionary of parameters

    Returns:
        train_sim, val_sim, test_sim (list) = List of simulations for each split.
    '''

    # Generate N-Body data
    if not os.path.exists(CONFIG['data_gen_path']):
        
        print(f"[Dataset] Generating dataset with {CONFIG['n_train']} training, {CONFIG['n_val']} validation and {CONFIG['n_test']} test simulations.")
        os.makedirs(os.path.dirname(CONFIG['data_gen_path']), exist_ok = True)

        # Generate the dataset
        train_sim = generateNBodyData(category = "train", config = data_gen_config)
        val_sim   = generateNBodyData(category = "val", config = data_gen_config)
        test_sim  = generateNBodyData(category = "test", config = data_gen_config)

        # Save the dataset
        with open(CONFIG['data_gen_path'], 'wb') as f:
            pickle.dump({
                'train_sim': train_sim,
                'val_sim': val_sim,
                'test_sim': test_sim
            }, f)
        print(f"[Dataset] Saved generated dataset to {CONFIG['data_gen_path']}.\n")

    else:

        print(f"[Dataset] Loading existing dataset from {CONFIG['data_gen_path']}.")

        # Load the dataset
        with open(CONFIG['data_gen_path'], 'rb') as f:
            generated_data = pickle.load(f)
            train_sim      = generated_data['train_sim']
            val_sim        = generated_data['val_sim']
            test_sim       = generated_data['test_sim']
        
        print(f"[Dataset] Loaded dataset with {len(train_sim)} training, {len(val_sim)} validation and {len(test_sim)} test simulations.\n")


    return train_sim, val_sim, test_sim
    



def main(CONFIG):
    '''
    Main function to run the N-Body experiment using EGNN paper

    Args:
        CONFIG (dict) = Configuration dictionary

    Returns
        -
    '''

    #$ Device setup
    device, device_ids, n_gpus = deviceSetup()
    print(f"\n\nDevice: {device} | Number of GPUs: {n_gpus}\n\n")

    # Dataset timer
    dataset_start = time.time()

    #$ Generate N-Body data
    data_gen_config = {
        'n_train'     : CONFIG['n_train'],
        'n_val'       : CONFIG['n_val'],
        'n_test'      : CONFIG['n_test'],
        'n_particles' : CONFIG['n_particles'],
        'n_steps'     : CONFIG['n_steps'],
        'dt'          : 0.001,
        'noise'       : 0.01
    }

    # Get the dataset splits
    train_sim, val_sim, test_sim = cacheDataset(data_gen_config)
    
    #$ Convert the generated data to PyTorch Geometric Data objects
    train_data = convertDatasetToPyG(train_sim)
    val_data   = convertDatasetToPyG(val_sim)
    test_data  = convertDatasetToPyG(test_sim)

    #$ DataLoader
    train_loader = DataLoader(train_data, batch_size = CONFIG['batch_size'], shuffle = True)
    val_loader   = DataLoader(val_data, batch_size = CONFIG['batch_size'], shuffle = False)
    test_loader  = DataLoader(test_data, batch_size = CONFIG['batch_size'], shuffle = False)

    # Print dataset stats
    dataset_end = time.time()
    print(f"\t\tDataset generation and conversion took {dataset_end - dataset_start:.2f} seconds.\n\n")


# VARIABLES
VERSION = "1"
CONFIG = {

    # Parameters based on paper (3000, 2000, 2000, 5, 1000)
    'n_train': 3000,
    'n_val': 2000,
    'n_test': 2000,
    'n_particles': 5,
    'n_steps': 1000,

    # Network parameters
    'hidden_dim': 64,
    'batch_size': 512,
    'epochs': 10000,
    'lr': 1e-4,
    'n_layers': 4,
    'criterion': nn.MSELoss(),

    # Misc
    'data_gen_path': None,
    'results_path': f"./csv/train_results_v{VERSION}.pkl",
    'best_egnn_path': f"./model/best_egnn_v{VERSION}.pth",
    'best_gnn_path': f"./model/best_gnn_v{VERSION}.pth",
    'plot_path': f"./plots/training_curves_v{VERSION}.png",
    'val_every': 10,
    'log_every': 5,
}

# Modify data_gen_path to the path where the generated data is stored
CONFIG['data_gen_path'] = f"./datanew/nbody_data_{CONFIG['n_train']}tr_{CONFIG['n_val']}v_{CONFIG['n_test']}te.pkl"
print(f"Dataset Path: {CONFIG['data_gen_path']}")


if __name__ == "__main__":

    # Main --> progression curves
    main(CONFIG)
