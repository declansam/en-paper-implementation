

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
warnings.filterwarnings("ignore", category = UserWarning, module = "torch_geometric.deprecation")

from torch_geometric.data import Data, DataLoader


##################################### SETUP #####################################

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



##################################### Dataset Generation #####################################

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

    # Get file name
    data_file_name = f"../datanew/nbody_data_{data_gen_config['n_train']}tr_{data_gen_config['n_val']}v_{data_gen_config['n_test']}te.pkl"

    # Generate N-Body data
    if not os.path.exists(data_file_name):
        
        print(f"[Dataset] Generating dataset with {data_gen_config['n_train']} training, {data_gen_config['n_val']} validation and {data_gen_config['n_test']} test simulations.")
        os.makedirs(os.path.dirname(data_file_name), exist_ok = True)

        # Generate the dataset
        train_sim = generateNBodyData(category = "train", config = data_gen_config)
        val_sim   = generateNBodyData(category = "val", config = data_gen_config)
        test_sim  = generateNBodyData(category = "test", config = data_gen_config)

        # Save the dataset
        with open(data_file_name, 'wb') as f:
            pickle.dump({
                'train_sim': train_sim,
                'val_sim': val_sim,
                'test_sim': test_sim
            }, f)
        print(f"[Dataset] Saved generated dataset to {data_file_name}.\n")

    else:

        print(f"[Dataset] Loading existing dataset from {data_file_name}.")

        # Load the dataset
        with open(data_file_name, 'rb') as f:
            generated_data = pickle.load(f)
            train_sim      = generated_data['train_sim']
            val_sim        = generated_data['val_sim']
            test_sim       = generated_data['test_sim']
        
        print(f"[Dataset] Loaded dataset with {len(train_sim)} training, {len(val_sim)} validation and {len(test_sim)} test simulations.\n")

    return train_sim, val_sim, test_sim





##################################### MODELS #####################################

##### 1) EGNN
class EGNNLayer(nn.Module):
    '''
    Class to define a single EGNN layer as mentioned in the paper
    '''

    def __init__(self, in_feat_dim, hidden_dim, out_feat_dim, act_func):
        '''
        Constructor for the EGNNLayer class

        Args:
            in_feat_dim (int)    = Number of input features
            hidden_dim (int)     = Hidden dimension
            out_feat_dim (int)   = Number of output features
            act_func (nn.Module) = Activation function

        Returns:
            -
        '''

        super(EGNNLayer, self).__init__()

        # Attributes
        self.epsilon        = 1e-05
        self.in_feat_dim    = in_feat_dim
        self.hidden_dim     = hidden_dim
        self.out_feat_dim   = out_feat_dim
        self.act_func       = act_func

        # In the paper, the only edge feature is the charge product so dim = 1
        self.edge_attr_dim  = 1

        # Equation (3): e(h_i, h_j, ||x_i - x_j||^2, a_ij)
        self.edge_in_dim = 2 * self.in_feat_dim + 1

        # If there are edge features, include them in message passing
        if self.edge_attr_dim > 0:
            self.edge_in_dim += self.edge_attr_dim

        # Equation (3)
        self.edge_mlp = self._edge_mlp()

        # Equation (4)
        self.coord_mlp = self._coord_mlp()

        # Equation (6)
        self.node_mlp = self._node_mlp()

        # Equation (7)
        self.momentum_vel_mlp = self._momentum_vel_mlp()

        # Section 3.3: Infer edge features for aggregation
        self.infer_edge_step = self._infer_edge_step()


    def forward(self, h, x, edge_index = None, edge_attr = None, v_init = None, charges = None):
        '''
        Forward pass for the EGNN layer

        Args:
            h (Tensor)          = Hidden state of nodes
            x (Tensor)          = Node coordinates
            edge_index (Long)   = Edge indices
            edge_attr (Tensor)  = Edge attributes
            v_init (Tensor)     = Initial velocity
            charges (Tensor)    = Charges
        
        Returns:
            h (Tensor) = Updated hidden state of nodes
            x (Tensor) = Updated node coordinates
            v (Tensor) = Updated velocity
        '''

        # Get number of nodes 
        n_nodes = h.size(0)

        #$ STEP 1
        # If there are no edge_index, create a fully connected graph
        # Implement Equation (5) for i =/ j
        if edge_index is None:
            edge_index = self._fcg(n_nodes, device = h.device)

        # Determine which nodes are connected
        row, col = edge_index

        #$ STEP 2
        # Edge features from charges
        if edge_attr is None and charges is not None:
            edge_attr = ( charges[row] * charges[col] ).unsqueeze(-1)

        #$ STEP 3
        # Calculate the distance between nodes
        dist_sq = torch.sum(
            (x[row] - x[col]) ** 2, dim = 1, keepdim = True
        )

        #$ STEP 4
        # Prepare edge inputs for equation (3)
        edge_inputs = [ h[row], h[col], dist_sq ]
        if edge_attr is not None:
            edge_inputs.append(edge_attr)

        edge_features = torch.cat(edge_inputs, dim = 1)

        #$ STEP 5: Edge Operation (using Equation (3))
        m_ij = self.edge_mlp(edge_features)

        #$ STEP 6: Use Equation (5) to infer edge features
        e_ij = self.infer_edge_step(m_ij)

        #$ STEP 7: Coordinate update using Equation (4)
        phi_x = self.coord_mlp(m_ij)
        x_diff = x[row] - x[col]                # (x_i - x_j)
        x_update = phi_x * x_diff               # phi_x * (x_i - x_j)

        #$ STEP 8: Equation (8)
        x_update = e_ij * x_update
        m_ij = e_ij * m_ij                      # e_ij * m_ij

        #$ STEP 9: Momentum update using Equation (7)
        if v_init is None:
            v_init = torch.zeros_like(x)

        # Update velocity
        vel_scale = self.momentum_vel_mlp(h)
        v_out = v_init * vel_scale

        #$ STEP 10: Aggregate messages for each node
        # x^{l+1}_i = x^l_i + v^{l+1}_i + C * Σ_{j =/ i} (x^l_i - x^l_j) * phi_x(m_ij)
        x_out = x.clone()
        for i in range(n_nodes):
            mask = row == i
            
            # In the paper, C = 1/(n_nodes - 1)
            if torch.any(mask):
                x_out[i] = x[i] + v_out[i] + torch.sum(x_update[mask], dim=0) / (n_nodes - 1)

        #$ STEP 11: Equation (5)
        # m_i = Σ_{j≠i} m_ij
        m_i = torch.zeros(n_nodes, self.hidden_dim, device = h.device)
        
        for i in range(n_nodes):
            mask = row == i
            if torch.any(mask):
                m_i[i] = torch.sum(m_ij[mask], dim = 0)

        #$ STEP 12: Node Operation using Equation (6)
        node_in = torch.cat( [h, m_i], dim = 1 )
        h_out = self.node_mlp(node_in)

        return h_out, x_out, v_out
        
    
    def _edge_mlp(self):
        '''
        Method to define the edge MLP for message passing in the EGNN layer
        Based on equation (3) in the paper
        '''

        edge_mlp = nn.Sequential(
            nn.Linear( self.edge_in_dim, self.hidden_dim ),
            self.act_func,
            nn.Linear( self.hidden_dim, self.hidden_dim ),
            self.act_func
        )

        return edge_mlp


    def _coord_mlp(self):
        '''
        Method to define the coordinate MLP for message passing in the EGNN layer
        Based on equation (4) in the paper
        '''

        coord_mlp = nn.Sequential(
            nn.Linear( self.hidden_dim, self.hidden_dim ),
            self.act_func,
            nn.Linear( self.hidden_dim, 1, bias = False )
        )

        return coord_mlp


    def _node_mlp(self):
        '''
        Method to define the node MLP for message passing in the EGNN layer
        Based on equation (6) in the paper
        '''

        node_mlp = nn.Sequential(
            nn.Linear( self.in_feat_dim + self.hidden_dim, self.hidden_dim ),
            self.act_func,
            nn.Linear( self.hidden_dim, self.out_feat_dim )
        )

        return node_mlp

    
    def _momentum_vel_mlp(self):
        '''
        Method to redefine the velocity update based on momentum 
        i.e. update Equation (4) in the paper to give Equation (7)
        '''

        vel_mlp = nn.Sequential(
            nn.Linear( self.in_feat_dim, self.hidden_dim ),
            self.act_func,
            nn.Linear( self.hidden_dim, 1, bias = False )
        )

        return vel_mlp


    def _infer_edge_step(self):
        '''
        Method to infer edge features for aggregation
        Based on Section 3.3 of the paper
        '''

        infer_edge_step = nn.Sequential(
            nn.Linear( self.hidden_dim, self.hidden_dim// 2 ),
            self.act_func,
            nn.Linear( self.hidden_dim// 2, 1),
            nn.Sigmoid()
        )

        return infer_edge_step


    def _fcg(self, n_nodes, device):
        '''
        Method to create a fully connected graph

        Args:
            n_nodes (int)         = Number of nodes
            device (torch.device) = Device to be used

        Returns:
            edge_index (Tensor) = Edge indices
        '''

        rows, cols = [], []

        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        
        edge_index = torch.tensor([rows, cols], dtype = torch.long, device = device)
        return edge_index



class EGNN(nn.Module):
    '''
    Class to define the EGNN model as mentioned in the paper
    '''

    def __init__(self, in_feat_dim, hidden_dim, out_feat_dim=None, n_layers=4):
        '''
        Constructor for the EGNN class

        Args:
            in_feat_dim (int)    = Number of input features
            hidden_dim (int)     = Hidden dimension
            out_feat_dim (int)   = Number of output features (defaults to in_feat_dim if None)
            n_layers (int)       = Number of EGNN layers

        Returns:
            -
        '''

        super(EGNN, self).__init__()

        # Attributes
        self.in_feat_dim  = in_feat_dim
        self.hidden_dim   = hidden_dim
        self.out_feat_dim = in_feat_dim if out_feat_dim is None else out_feat_dim

        # Activation function
        self.act_func = nn.ReLU()

        #$ 1. Linear layer to map velocity norms as input features
        self.vel_norm_mlp = nn.Linear(
            1,
            self.in_feat_dim
        )

        #$ 2. EGNN layers
        self.layers = nn.ModuleList()

        # 2a. First layer
        self.layers.append(
            EGNNLayer(
                in_feat_dim = self.in_feat_dim,
                hidden_dim = self.hidden_dim,
                out_feat_dim = self.hidden_dim,
                act_func = self.act_func
            )
        )

        # 2b. Intermediate layers
        for _ in range(n_layers - 2):
            self.layers.append(
                EGNNLayer(
                    in_feat_dim = self.hidden_dim,
                    hidden_dim = self.hidden_dim,
                    out_feat_dim = self.hidden_dim,
                    act_func = self.act_func
                )
            )

        # 2c. Last layer
        self.layers.append(
            EGNNLayer(
                in_feat_dim = self.hidden_dim,
                hidden_dim = self.hidden_dim,
                out_feat_dim = self.out_feat_dim,
                act_func = self.act_func
            )
        )


    def forward(self, vel_norm, x, v, edge_index = None, charges = None):
        '''
        Forward pass for the EGNN model

        Args:
            vel_norm (Tensor)   = Velocity norms
            x (Tensor)          = Node coordinates
            v (Tensor)          = Initial velocity
            edge_index (Long)   = Edge indices
            charges (Tensor)    = Charges

        Returns:
            h (Tensor) = Updated hidden state of nodes
            x (Tensor) = Updated node coordinates
            v (Tensor) = Updated velocity
        '''

        # STEP 1: Map velocity norms to input features
        h = self.vel_norm_mlp(vel_norm)

        # STEP 2: Loop through the EGNN layers
        for layer in self.layers:
            h, x, v = layer(
                        h = h, 
                        x = x, 
                        edge_index = edge_index,
                        edge_attr = None,
                        v_init = v, 
                        charges = charges
            )

        return h, x, v





##### 2. GNN
class GNN(nn.Module):
    '''
    Standard GNN baseline (non-equivariant)
    '''
    
    def __init__(self, in_node_features, hidden_dim = 64, num_layers = 4):
        '''
        Constructor for the GNN class
        
        Args:
            in_node_features (int) = Number of input node features
            hidden_dim (int)       = Hidden dimension size
            num_layers (int)       = Number of message passing layers
            
        Returns:
            -
        '''
        
        super(GNN, self).__init__()

        # Attributes
        self.in_node_features = in_node_features
        self.hidden_dim       = hidden_dim
        self.num_layers       = num_layers
        self.edge_attr_dim    = 1
        self.epsilon          = 1e-05
        self.act_func         = nn.SiLU()

        # Edge attribute (since it's just charge product, dim = 1)
        self.edge_attr_dim = 1

        # Position coordinates are included in the initial node features
        self.pos_dim = 3
        
        #$ STEP 1: Velocity norm encoder (as per paper)
        self.velocity_encoder = nn.Linear(1, in_node_features)

        #$ STEP 2: Initial node embedding MLP
        self.node_mlps = nn.ModuleList()
        self.node_mlps.append( self._node_mlp() )

        #$ STEP 3: Edge and message MLPs for each layer
        self.edge_mlps = nn.ModuleList()
        self.message_mlps = nn.ModuleList()

        for _ in range(num_layers):
            
            # Edge operation MLPs
            self.edge_input_dim = hidden_dim * 2
            if self.edge_attr_dim > 0:
                self.edge_input_dim += self.edge_attr_dim
            
            # Add edge MLPs
            self.edge_mlps.append( self._edge_mlp() )

            # Node update MLPs
            self.message_mlps.append( self._msg_mlp() )

        #$ STEP 4: Final prediction layer (position output)
        self.out_mlp = self._out_mlp()




    def forward(self, vel_norms, x, edge_index = None, charges = None):
        '''
        Forward pass for the GNN model
        
        Args:
            vel_norms (Tensor)   = Velocity norms [num_nodes, 1]
            x (Tensor)           = Node coordinates [num_nodes, 3]
            edge_index (Tensor)  = Optional edge indices [2, num_edges]
            charges (Tensor)     = Optional node charges [num_nodes]
            
        Returns:
            position_pred (Tensor) = Predicted positions [num_nodes, 3]
        '''
        
        # Get number of nodes
        n_nodes = vel_norms.size(0)

        #$ STEP 1: Create fully connected edges if not provided
        if edge_index is None:
            edge_index = self._fcg(n_nodes, device = vel_norms.device)

        # get row and col indices
        row, col = edge_index[0], edge_index[1]
        

        #$ STEP 2: Create edge attributes from charges if provided
        edge_attr = None
        if charges is not None:
            
            # Compute charge products for each edge
            edge_attr = (charges[row] * charges[col]).unsqueeze(-1)



        #$ STEP 3: Initial node embedding
        h = self.velocity_encoder(vel_norms)
        h = torch.cat([h, x], dim = 1)
        h = self.node_mlps[0](h)


        #$ STEP 4: Message passing through all layers
        for i in range(self.num_layers):
            
            # STEP 4a: Compute edge features and messages
            edge_features = torch.cat([h[row], h[col]], dim = 1)
            if edge_attr is not None:
                edge_features = torch.cat([edge_features, edge_attr], dim = 1)
                
            m_ij = self.edge_mlps[i](edge_features)

            # STEP 4b: Aggregate messages for each node
            m_i = torch.zeros(n_nodes, self.hidden_dim, device = h.device)
            for j in range(n_nodes):
                
                # Mask to select edges connected to node j
                mask = row == j

                # Use mask to aggregate messages
                if torch.any(mask):
                    m_i[j] = torch.sum(m_ij[mask], dim = 0)

            # STEP 4c: Update node embeddings using aggregated messages
            node_inputs = torch.cat([h, m_i], dim = 1)
            h = self.message_mlps[i](node_inputs)



        #$ STEP 5: Final position prediction
        position_pred = self.out_mlp(h)
        
        return position_pred




    def _node_mlp(self):
        '''
        Method to define the node MLP for message passing in the GNN layer
        '''

        node_mlp = nn.Sequential(
                nn.Linear(self.in_node_features + self.pos_dim, self.hidden_dim),
                self.act_func
        )

        return node_mlp




    def _edge_mlp(self):
        '''
        Method to define the edge MLP for message passing in the GNN layer
        '''

        edge_mlp = nn.Sequential(
                    nn.Linear(self.edge_input_dim, self.hidden_dim),
                    self.act_func,
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    self.act_func
        )
        return edge_mlp


    
    def _msg_mlp(self):
        '''
        Method to define the message MLP for message passing in the GNN layer
        '''

        msg_mlp = nn.Sequential(
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                    self.act_func,
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    self.act_func
        )

        return msg_mlp

    

    def _out_mlp(self):
        '''
        Method to define the output MLP for message passing in the GNN layer
        '''

        out_mlp = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    self.act_func,
                    nn.Linear(self.hidden_dim, self.pos_dim)
        )

        return out_mlp


    def _fcg(self, n_nodes, device):
        '''
        Method to create a fully connected graph
        
        Args:
            n_nodes (int)         = Number of nodes
            device (torch.device) = Device to be used
            
        Returns:
            edge_index (Tensor) = Edge indices [2, num_edges]
        '''
        
        rows, cols = [], []

        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        
        edge_index = torch.tensor([rows, cols], dtype = torch.long, device = device)
        return edge_index






##################################### Trainer + Eval #####################################
def tester(model, loader, criterion, device, model_name = "egnn"):
    '''
    Function to test the model on the validation dataset

    Args: 
        model (nn.Module)       = Model to be tested
        loader (DataLoader)     = DataLoader for the validation dataset
        criterion (nn.Module)   = Loss function
        device (torch.device)   = Device to be used
        model_name (str)        = Name of the model
    
    Returns:
        loss (float) = Average loss on the validation dataset
    '''

    # Set to eval mode
    model.eval()

    # Init parameters
    total_loss, n_batches = 0, 0

    # Loop through the validation dataset
    with torch.no_grad():
        for data in loader:

            # Send data to device
            data = data.to(device)
            x_pred = None

            # Select model
            if model_name == "egnn":
                _, x_pred, _ = model(
                                vel_norm = data.v_norm,
                                x = data.x,
                                v = data.v,
                                edge_index = None,
                                charges = data.c
                )

            elif model_name == "gnn":
                x_pred = model(
                                vel_norms = data.v_norm,
                                x = data.x,
                                edge_index = None,
                                charges = data.c
                )

        
            # Calculate loss
            loss = criterion(x_pred, data.y)

            # Compute loss and n_batches
            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def trainer(model, loader, optimizer, criterion, device, model_name = "egnn"):
    '''
    Function to train the model on the training dataset

    Args:
        model (nn.Module)       = Model to be trained
        loader (DataLoader)     = DataLoader for the training dataset
        optimizer (torch.optim) = Optimizer
        criterion (nn.Module)   = Loss function
        device (torch.device)   = Device to be used
        model_name (str)        = Name of the model

    Returns:
        loss (float) = Average loss on the training dataset
    '''

    # Set to train mode
    model.train()

    # Init parameters
    total_loss, n_batches = 0, 0

    # Loop through the training dataset
    for data in loader:

        # Send data to device
        data = data.to(device)
        x_pred = None

        # Zero grad
        optimizer.zero_grad()

        # Select appropriate model
        if model_name == "egnn":
            _, x_pred, _ = model(
                            vel_norm = data.v_norm,
                            x = data.x,
                            v = data.v,
                            edge_index = None,
                            charges = data.c )

        elif model_name == "gnn":
            x_pred = model(
                            vel_norms = data.v_norm,
                            x = data.x,
                            edge_index = None,
                            charges = data.c )

        # Calculate loss
        loss = criterion(x_pred, data.y)

        # Backprop
        loss.backward()
        optimizer.step()

        # Compute loss and n_batches
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches




##################################### Visualization #####################################

def plotTrainingCurves(train_losses, val_losses, model_names, save_path):
    '''
    Function to plot the training curves

    Args:
        train_losses (list) = List of training losses
        val_losses (list)   = List of validation losses
        model_names (list)  = List of model names
        save_path (str)     = Path to save the plot
    
    Returns:
        -
    '''

    #$ Setup figure
    plt.figure(figsize = (12, 8))
    
    #$ STEP 1: Plot training losses
    for m in model_names:
        if m in train_losses and len(train_losses[m]) > 0:
            epochs = range(1, len(train_losses[m]) + 1)
            plt.plot(epochs, train_losses[m], '-', label = f'{m} (Train)')
    
    #$ STEP 2: Plot validation losses
    for m in model_names:
        if m in val_losses and len(val_losses[m]) > 0:
            
            # For validation points --> less number of points
            val_epochs = [i * CONFIG['val_every'] for i in range(1, len(val_losses[m]) + 1)]
            plt.plot(val_epochs, val_losses[m], 'o-', label = f'{m} (Val)')
    
    #$ STEP 3: Setup plot
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.grid(True, which = "both", ls = "--")
    plt.legend()
    
    #$ STEP 4: Save and display
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}.")



def plotMSEvsSampleCurves(sample_sizes, losses, model_names, save_path):
    '''
    Function to plot the MSE vs Sample size curves

    Args:
        sample_sizes (list)  = List of sample sizes
        losses (list)        = List of losses
        model_names (list)   = List of model names
        save_path (str)      = Path to save the plot
    
    Returns:
        -
    '''

    #$ Setup figure
    plt.figure(figsize = (12, 8))
    markers = ['o', 's']
    
    #$ STEP 1: Plot test losses for each model
    for i, m in enumerate(model_names):
        
        if m in losses and len(losses[m]) > 0:
            plt.plot(sample_sizes, losses[m], 
                    marker = markers[i % len(markers)], 
                    linewidth = 2, 
                    markersize = 8,
                    label = m)
    
    #$ STEP 2: Setup plot
    plt.title('MSE Loss vs. Number of Training Samples', fontsize = 16)
    plt.xlabel('Number of Training Samples', fontsize = 14)
    plt.ylabel('Mean Squared Error (Test Set)', fontsize = 14)
    plt.grid(True, which = "both", ls = "--")
    plt.legend(fontsize = 12)
    
    #$ STEP 3: Save and display
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}.")





##################################### MAIN (Fixed Samples) #####################################

def main(CONFIG, sample_run = False, sample_dict = None):
    '''
    Main function to run the N-Body experiment using EGNN paper

    Args:
        CONFIG (dict)       = Configuration dictionary
        sample_run (bool)   = Boolean that determines if the run is a sample run
        sample_dict (dict)  = Dictionary of sample configuration

    Returns
        -
    '''

    # Logging
    if sample_run:
        print(f"\n\n{'='*20} Running experiments with {sample_dict['n_train']} Sample Runs {'='*20}\n\n")
    else:
        print(f"\n\n{'='*20} Running experiment {'='*20}\n\n")

    #$ Device setup
    device, device_ids, n_gpus = deviceSetup()
    print(f"Device: {device} | Number of GPUs: {n_gpus}\n\n")

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

    # In case of sample run, update the number of simulations
    if sample_run:
        data_gen_config['n_train'] = sample_dict['n_train']
        data_gen_config['n_val']   = sample_dict['n_val']
        data_gen_config['n_test']  = sample_dict['n_test']

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

    # Logging
    print(f"\tDataset splits: Train: {len(train_data)} | Validation: {len(val_data)} | Test: {len(test_data)}")

    # Print dataset stats
    dataset_end = time.time()
    print(f"\tDataset generation and conversion took {dataset_end - dataset_start:.2f} seconds.\n\n")


    #$ Model setup
    print(f"Model setup with {CONFIG['n_layers']} EGNN layers.")
    in_feat_dim  = CONFIG['hidden_dim']

    # Model
    # in_feat_dim, hidden_dim, out_feat_dim, n_layers
    egnn_model = EGNN(
        in_feat_dim = in_feat_dim,
        hidden_dim = CONFIG['hidden_dim'],
        out_feat_dim = in_feat_dim,
        n_layers = CONFIG['n_layers']
    )
    egnn_model = egnn_model.to(device)

    # GNN Model
    gnn_model = GNN(
        in_node_features = in_feat_dim,
        hidden_dim = CONFIG['hidden_dim'],
        num_layers = CONFIG['n_layers']
    )
    gnn_model = gnn_model.to(device)

    # If multi-GPU, wrap the model
    if n_gpus > 1:
        egnn_model = nn.DataParallel(egnn_model, device_ids = device_ids)
        gnn_model = nn.DataParallel(gnn_model, device_ids = device_ids)

    # Optimizers
    egnn_optimizer = torch.optim.Adam(egnn_model.parameters(), lr = CONFIG['lr'])
    gnn_optimizer = torch.optim.Adam(gnn_model.parameters(), lr = CONFIG['lr'])

    # Loss tracking
    train_loss = {
        'EGNN': [],
        'GNN': []
    }

    val_loss = {
        'EGNN': [],
        'GNN': []
    }


    #$ TRAINING
    print(f"\n\n{'='*15} Training in progress {'='*15} \n\n")
    best_egnn_loss = float('inf')
    best_gnn_loss  = float('inf')

    # Timer start
    trainer_start = time.time()

    # based on the type of experiment, change the description
    tqdm_desc = f"Training {sample_dict['n_train']} samples" if sample_run else "Training progress"

    # Loop through the number of epochs
    for e in tqdm( range(CONFIG['epochs']), desc = tqdm_desc):

        # Train the EGNN model
        egnn_loss = trainer(egnn_model, train_loader, egnn_optimizer, CONFIG['criterion'], device, 'egnn')
        train_loss['EGNN'].append(egnn_loss)

        # Train the GNN model
        gnn_loss = trainer(gnn_model, train_loader, gnn_optimizer, CONFIG['criterion'], device, 'gnn')
        train_loss['GNN'].append(gnn_loss)

        # Log the training loss
        if (e + 1) % CONFIG['log_every'] == 0:
            time_spent = time.time() - trainer_start
            time_rem   = (time_spent / (e + 1)) * (CONFIG['epochs'] - e)
            print(f"\n\tEpoch: {e + 1}/{CONFIG['epochs']} | Time taken: {time_spent:.2f} seconds | Time remaining: {time_rem:.2f} seconds")
            print(f"\n\tEGNN Loss: {egnn_loss:.4f} | GNN Loss: {gnn_loss:.4f}\n")

        # Test the EGNN model every 'val_every' epochs
        if (e + 1) % CONFIG['val_every'] == 0:

            print(f"\t\tValidating models at epoch {e + 1}")

            # Validate EGNN
            egnn_val_loss = tester(egnn_model, val_loader, CONFIG['criterion'], device, 'egnn')
            val_loss['EGNN'].append(egnn_val_loss)

            # Validate GNN
            gnn_val_loss = tester(gnn_model, val_loader, CONFIG['criterion'], device, 'gnn')
            val_loss['GNN'].append(gnn_val_loss)

            # Save the best models
            if sample_run:
                egnn_name = f"{CONFIG['best_egnn_path']}_sample_{sample_dict['n_train']}.pth"
                gnn_name  = f"{CONFIG['best_gnn_path']}_sample_{sample_dict['n_train']}.pth"
            else:
                egnn_name = f"{CONFIG['best_egnn_path']}.pth"
                gnn_name  = f"{CONFIG['best_gnn_path']}.pth"
                
            if egnn_val_loss < best_egnn_loss:
                best_egnn_loss = egnn_val_loss
                torch.save(egnn_model.state_dict(), egnn_name)

            if gnn_val_loss < best_gnn_loss:
                best_gnn_loss = gnn_val_loss
                torch.save(gnn_model.state_dict(), gnn_name)

            print(f"\t\tEGNN Validation Loss: {egnn_val_loss:.4f} | GNN Validation Loss: {gnn_val_loss:.4f}\n")

    # Timer end
    trainer_end = time.time()
    print(f"Training completed in {trainer_end - trainer_start:.2f} seconds.\n\n")

    #$ FINAL TESTING
    print(f"\n\n{'='*15} Testing in progress {'='*15} \n\n")
    
    egnn_final_loss = tester(egnn_model, test_loader, CONFIG['criterion'], device, 'egnn')
    gnn_final_loss = tester(gnn_model, test_loader, CONFIG['criterion'], device, 'gnn')
    
    print("Test results:")
    print(f"\tEGNN: {egnn_final_loss:.4f}")
    print(f"\tGNN:  {gnn_final_loss:.4f}\n\n")
    print("Experiment completed!\n\n")


    #$ Plot (only in non-sample case)
    if sample_run:
        loss_curve_name = f"{CONFIG['plot_path']}_sample_{data_gen_config['n_train']}.png"
    else:
        loss_curve_name = f"{CONFIG['plot_path']}.png"
        
    plotTrainingCurves(
                    train_loss, val_loss, 
                    ['EGNN', 'GNN'], loss_curve_name      
    )

    #$ Save the results
    final_results = {
        'egnn_test_loss': egnn_final_loss,
        'gnn_test_loss': gnn_final_loss,
        'best_egnn_loss': best_egnn_loss,
        'best_gnn_loss': best_gnn_loss,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'total_time': time.time() - dataset_start
    }

    if sample_run:
        results_file_name = f"{CONFIG['results_path']}_sample_{sample_dict['n_train']}.pkl"
    else:
        results_file_name = f"{CONFIG['results_path']}.pkl"
        
    with open( results_file_name, 'wb' ) as f:
        pickle.dump(final_results, f)
    print(f"Results saved to {results_file_name}.")


    #$ Save the model
    if sample_run:
        egnn_name_final = f"{CONFIG['best_egnn_path']}_sample{sample_dict['n_train']}_final.pth"
        gnn_name_final  = f"{CONFIG['best_gnn_path']}_sample{sample_dict['n_train']}_final.pth"
    else:
        egnn_name_final = f"{CONFIG['best_egnn_path']}_final.pth"
        gnn_name_final  = f"{CONFIG['best_gnn_path']}_final.pth"
        
    torch.save(egnn_model.state_dict(), egnn_name_final)
    torch.save(gnn_model.state_dict(), gnn_name_final)
    print(f"Final models saved! \n\n")

    #$ Return
    return final_results




##################################### Variable Sapmles #####################################
def sampleRunMain(CONFIG):
    '''
    Function to run experiments with different numbers of training samples
    
    Args:
        CONFIG (dict)  = Configuration dictionary
    
    Returns:
        results (dict) = Dictionary containing experiment results
    '''
    
    #$ STEP 1: Define sample sizes and epoch configurations
    sample_sizes = [20, 100, 500, 1000, 2500]
    
    # Size: Epochs
    epochs_per_size = {
        20: 700,
        100: 700,
        500: 600,
        1000: 500,
        2500: 400
    }
    
    #$ STEP 2: Initialize results dictionary
    results = {
        'sample_sizes': sample_sizes,
        'losses': {'EGNN': [], 'GNN': [] }
    }
    
    #$ STEP 3: Run experiments for each sample size
    for n_samples in sample_sizes:

        # Update epochs before running
        epochs = epochs_per_size.get(n_samples, 1000)
        CONFIG['epochs'] = epochs
        
        main_results = main(
                            CONFIG, 
                            sample_run = True, 
                            sample_dict = {
                                'n_train': n_samples,
                                'n_val': 100,
                                'n_test': 100
                            }
        )
        
        # Save test losses
        results['losses']['EGNN'].append( main_results['egnn_test_loss'] )
        results['losses']['GNN'].append( main_results['gnn_test_loss'] )
        
        # Save files
        file_name_sample = f"{CONFIG['sample_results_path']}_sample_{n_samples}.pkl"
        with open(file_name_sample, "wb") as f:
            pickle.dump(results, f)
    
    # Logging
    print(f"\n\n{'=-'*10} Samples-based experiment completed. {'=-'*10}")
    
    #$ STEP 4: Generate final plot
    plotMSEvsSampleCurves(
        results['sample_sizes'], 
        results['losses'], 
        ['EGNN', 'GNN'], 
        save_path = CONFIG['sample_plot_path']
    )
    
    # Print results based on sample sizes
    for i, n_samples in enumerate(results['sample_sizes']):
        print(f"\nResults for {n_samples} samples:")
        
        for m in ['EGNN', 'GNN']:
            print(f"  {m}: {results['losses'][m][i]:.6f}")
    
    return results





##################################### CONFIG #####################################

VERSION = "3"
CONFIG = {

    # Parameters based on paper (3000, 2000, 2000, 5, 1000)
    'n_train': 3000,
    'n_val': 2000,
    'n_test': 2000,
    'n_particles': 5,
    'n_steps': 1000,

    # Network parameters
    'hidden_dim': 64,
    'batch_size': 256,
    'epochs': 800,
    'lr': 1e-4,
    'n_layers': 4,
    'criterion': nn.MSELoss(),

    # Misc
    'results_path': f"./csv/train_results_v{VERSION}",
    'sample_results_path': f"./csv/sample_results_v{VERSION}",

    'best_egnn_path': f"./model/best_egnn_v{VERSION}",
    'best_gnn_path': f"./model/best_gnn_v{VERSION}",

    'plot_path': f"./plots/training_curves_v{VERSION}",
    'sample_plot_path': f"./plots/mse_curves_v{VERSION}.png",

    'val_every': 10,
    'log_every': 5,
}



if __name__ == "__main__":

    # Make sure the directories exist
    os.makedirs('./csv', exist_ok = True)
    os.makedirs('./datanew', exist_ok = True)
    os.makedirs('./model', exist_ok = True)
    os.makedirs('./plots', exist_ok = True)

    # OPTION 1 (Progression Curves - Fixed Experiment)
    # _ = main(CONFIG)
    
    # OPTION 2 (MSE Curves - Variable Number of Samples)
    _ = sampleRunMain(CONFIG)

