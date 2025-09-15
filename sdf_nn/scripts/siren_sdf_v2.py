#!/usr/bin/env python3
# INFORMACIÓN SOBRE EL CÓDIGO
# Esta es la adaptación de la SIREN al repo del HIO_SDF. El objetivo era no hacer el launch del hio_sdf y utilizar este código. El problema que tiene es que, como /tf no es temporizado,
# no se puede utilizar el AsynchronousSynchronizer que venía usando hasta el momento, por lo que se debería cambiar (por ejemplo, tiene que recibir siempre primero una posición antes de empezar con lo otro)

import rospy
from std_msgs.msg import String
import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Imu
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped
from numpy import random
from scipy.spatial import cKDTree
import message_filters
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from voxblox_msgs.msg import DiscreteSDF
import struct
from tf2_msgs.msg import TFMessage
#import time

#import os
import torch
import torch.nn as nn
#import torch.optim as optim
#from siren_pytorch import SirenNet
#from torch.utils.data import DataLoader, TensorDataset
#from torchvision import datasets, transformsimport struct
f_training_points = 0
f_certain_training_points = 0
tp_print_limit = 1.9
f_sdf_tp_distribution = 0


# ======== VARIABLES ========

total_wall_point_list = np.empty((0,3)) #Complete point list
training_positions = np.empty((0,3)) # List of drone positions where training was done
training_points = np.empty((0,4)) # Training points list
num_topics = 0 # Topic selection variable
updated_points_counter = 0 # Updated points counter
iters = 0 # Number of training iterations already done

Q = np.array([1, 0, 0, 0])
Q_last = np.array([1, 0, 0, 0])
x_pos = 0
y_pos = 0
z_pos = 0
x_last = -1000
y_last = -1000
z_last = -1000
x_ini = np.nan
y_ini = np.nan
z_ini = np.nan

#Voxel map and functions definition 
#voxel_size = 0.2 # voxel size (m)
#voxel_map_dim = 5 # voxel map radius (m) 
#voxel_grid_dim = int(voxel_map_dim / voxel_size) # voxel map radius (in voxel numbers)
#map_total_dim = int(2*voxel_grid_dim + 1) # voxel map dimension (in voxel numbers)
#voxel_grid = np.full((map_total_dim,map_total_dim, map_total_dim), -1000.1)
#total_points = map_total_dim**3

#occupancy_grid = np.full((map_total_dim,map_total_dim, map_total_dim), 0)

# ========= VOXBLOX DECLARATIONS =======

esdf_layer_msg_v = None
voxel_size = 0.1 # IMPORTANT: Make sure this matches your Voxfield configuration
voxel_per_side = 8 # IMPORTANT: Make sure this matches your Voxfield configuration
kEpsilon = 0.000001
kNumDataPacketsPerVoxel = 2
block_size = voxel_per_side * voxel_size

num_voxfield_samples_surface: 10000 # Max number of used Voxfield samples lying close to the surface for each training iteration
num_voxfield_samples_freespace: 30000 # Max number of used Voxfield samples away from the surface for each training iteration

# Preparing Voxfield meshgrid
nx, ny, nz = np.meshgrid(
    np.linspace(0, voxel_per_side-1, voxel_per_side),
    np.linspace(0, voxel_per_side-1, voxel_per_side),
    np.linspace(0, voxel_per_side-1, voxel_per_side)
)
meshgrid = np.zeros((voxel_per_side**3, 3))
meshgrid[:, 0] = np.transpose(nx, (2,0,1)).flatten() 
meshgrid[:, 1] = np.transpose(ny, (2,0,1)).flatten()
meshgrid[:, 2] = np.transpose(nz, (2,0,1)).flatten()
meshgrid = (meshgrid + 0.5) * voxel_size


# ======== FUNCTIONS ========


def RotQuad(Q): #Puede estar al revés (comprobar)
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
    
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
    
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
    
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix

def check_line_of_sight(x_ini,y_ini,z_ini,x_last,y_last,z_last): # Returns 1 if line of sight exists, 0 otherwise

    global voxel_grid

    # Calculte difference between points
    dx = x_last - x_ini
    dy = y_last - y_ini
    dz = z_last - z_ini

    # Determne increments for each axis
    sx = 1 if dx > 0 else -1
    sy = 1 if dy > 0 else -1
    sz = 1 if dz > 0 else -1

    dx = abs(dx)
    dy = abs(dy)
    dz = abs(dz)

    # Initialize internal counters
    cont_x = 1
    cont_y = 1
    cont_z = 1

    x_act = x_ini
    y_act = y_ini
    z_act = z_ini

    while (x_act, y_act, z_act) != (x_last, y_last, z_last):
        if (dx != 0):
            x_coef = cont_x/(dx*2)
        else:
            x_coef = 1.1
        if (dy != 0):
            y_coef = cont_y/(dy*2)
        else:
            y_coef = 1.1
        if (dz != 0):
            z_coef = cont_z/(dz*2)
        else:
            z_coef = 1.1

        if (x_coef <= y_coef and x_coef <= z_coef):
            x_act += sx
            cont_x += 2
        if (y_coef <= x_coef and y_coef <= z_coef):
            y_act += sy
            cont_y += 2
        if (z_coef <= y_coef and z_coef <= x_coef):
            z_act += sz
            cont_z += 2

        if (voxel_grid[x_act][y_act][z_act] == 0):
            return 0
    
    return 1

def update_sdf_point(target_point_x, target_point_y, target_point_z, x_drone, y_drone, z_drone, discrete_wall_coordinates):
    # Comparar cada punto con los obstáculos nuevos y actualizarlo si el valor absoluto es menor con las nuevas paredes
    global voxel_grid, updated_points_counter, total_points
    sdf_prov_distance = np.nan # Inicializo la variable para guardar el sdf
    for row in discrete_wall_coordinates: # Por cada nuevo obstáculo detectado
        dist_to_obstacle = np.sqrt((target_point_x - row[0])**2 + (target_point_y - row[1])**2 + (target_point_z - row[2])**2) * voxel_size
        if (np.isnan(sdf_prov_distance) or dist_to_obstacle < sdf_prov_distance): # Si es el primer punto o el nuevo punto está más cerca que los anteriores
            sdf_prov_distance = dist_to_obstacle # Actualizo el sdf estimado

    # Una vez terminado se comprueba el signo
    if(voxel_grid[target_point_x][target_point_y][target_point_z] <= 0): # Si el sdf anterior es negativo
        los_result = check_line_of_sight(x_drone,y_drone,z_drone,target_point_x, target_point_y, target_point_z) # Checkea Line of Sight
        if(los_result == 1): # Si hay línea de visión
            voxel_grid[target_point_x][target_point_y][target_point_z] = sdf_prov_distance # El nuevo sdf es positivo
        else:                 # Si no
            voxel_grid[target_point_x][target_point_y][target_point_z] = -sdf_prov_distance # El nuevo sdf es negativo
    else:
        voxel_grid[target_point_x][target_point_y][target_point_z] = sdf_prov_distance # Si ya era positivo antes, debe seguir siéndolo
    
    updated_points_counter += 1
    print("Updated points: ", updated_points_counter, "/", total_points, '|| Point updated to:', voxel_grid[target_point_x,target_point_y,target_point_z])
                
def update_sdf(x_drone,y_drone,z_drone, discrete_wall_coordinates):
    global voxel_grid, updated_points_counter
    updated_points_counter = 0
    for i in range(voxel_grid.shape[0]):
        for j in range(voxel_grid.shape[1]):
            for k in range(voxel_grid.shape[2]):
                if(voxel_grid[i][j][k] != 0):
                    update_sdf_point(i,j,k,x_drone,y_drone,z_drone, discrete_wall_coordinates)
    print("SDF Updated Successfully")

def update_occupancy(discrete_wall_coordinates):
    global occupancy_grid
    for row in discrete_wall_coordinates:
        occupancy_grid[int(row[0])][int(row[1])][int(row[2])] = 1



# ======== SIREN NEURAL NETWORK ======== (4 hidden layers with 256 neurones, periodic (sinusoidal) activations, linear output layer, custom loss, custom initial weights)

dtype = torch.float
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(10 * input)

class FCNetWork(nn.Module): # Fully connected generic neural network definition (can be expanded with new nonlinearities)
    def __init__(
            self,
            in_features,
            out_features,
            n_hidden_layers,
            hidden_features,
            outermost_linear = False,
            nonlinear_type = 'relu',
            weight_init_mode = None
        ):
        super().__init__()

        self.first_layer_init = None

        # Dictionary for nonlinearity types (including weight init and, if needed, first-layer init)

        nl_inits = {'sine':(Sine(), sine_init, None),
                    'relu':(nn.ReLU(inplace=True), init_weights_normal, None)}
        
        nl, nl_weight_init, fl_init = nl_inits[nonlinear_type]

        if weight_init_mode is not None:
            self.weight_init_mode = weight_init_mode
        else:
            self.weight_init_mode = nl_weight_init
        
        # Net construction
        
        self.net = [] # Starts with an empty net
        self.net.append(nn.Sequential(nn.Linear(in_features, hidden_features), nl)) # Adds the first layers (linear + nonlinear activation)

        for i in range(n_hidden_layers):        # Adds intermediate layers
            self.net.append(nn.Sequential(nn.Linear(hidden_features, hidden_features), nl))

        if outermost_linear:            # Adds the last layer (linear or with a nonlinear activation)
            self.net.append(nn.Sequential(nn.Linear(hidden_features, out_features)))
        else:
            self.net.append(nn.Sequential(nn.Linear(hidden_features, out_features), nl))

        self.net = nn.Sequential(*self.net) # Wraps the net into a Sequential container

        # Applies the desired weight initialization (if defined)

        if self.weight_init_mode is not None:
            self.net.apply(self.weight_init_mode)

        # Applies first layer initialization (if defined)

        if fl_init is not None:
            self.net[0].apply(fl_init)
    
    def forward(self, input):
        return self.net(input)

# ---INITIALIZATION METHODS----
    
def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

def sine_init(m):
    with torch.no_grad(): # Deactivates the auto_grad tracking for this operation (needed if using autograd later in the code)
        if hasattr(m, 'weight'):
            n_inputs = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6/n_inputs)/10, np.sqrt(6/n_inputs)/10)
    
# ----------------------------
    
class SIREN(nn.Module):

    def __init__(
            self,
            in_features = 3,
            out_features = 1,
            type = 'sine',
            hidden_features = 256,
            n_hidden_layers = 4,
        ):
        super().__init__()

        self.net = FCNetWork(
            in_features = in_features,
            out_features = out_features,
            n_hidden_layers = n_hidden_layers,
            hidden_features = hidden_features,
            outermost_linear = True,
            nonlinear_type = type
        )

        print(self)
    
    def forward(self, input):
        return self.net(input)

# Create the model

#siren_model = SIREN().to(device)

# ========= TRAINING DEFINITIONS =========
class Dataset(torch.utils.data.Dataset): # Structure to save collected data
    def __init__(self, data, dtype = torch.float, device = 'cuda'):
        self.dtype = dtype
        self.device = device
        self.pc_data = data[:, :3]   # Pointcloud info
        self.sdf_data = data[:, 3]  # SDF
    
    def __getitem__(self, index):
        point = self.pc_data[index,:]
        sdf = self.sdf_data[index]
        return point, sdf
    
    def __len__(self):
        return len(self.pc_data)
    
    def update_dataset(self, data):
        self.pc_data = data[:, :3]   # Pointcloud info
        self.sdf_data = data[:, 3]  # SDF

def sdf_loss(sdf, target_sdf):
    return torch.abs(sdf-target_sdf)

def eikonal_loss(grad_sdf):
    return torch.abs(torch.norm(grad_sdf, dim= -1, keepdim= True) - 1)

class SDFTrainer(): # Trainer class that gets called during training
    def __init__(
        self,
        learning_rate = 0.0004,
        weight_decay = 0.012,
        sdf_loss_weight = 5.0,
        eikonal_loss_weight = 2.0,
        device = 'cuda',
        dtype = torch.float,
        batch_size = 64
    ):
        # Device
        self.device = device
        self.dtype = dtype

        # Optimization params
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Loss params
        self.sdf_loss_weight = sdf_loss_weight
        self.eikonal_loss_weight = eikonal_loss_weight

        # Model declaration
        self.model = SIREN()
        self.model.to(self.device)
        self.model.train()

        # Dataset
        init_data_rand = torch.rand(1000,7)
        self.dataset = Dataset(init_data_rand, dtype = self.dtype, device = self.device)
        self.batch_size = batch_size


        # Training optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr= self.learning_rate, weight_decay= self.weight_decay)

        # Loss init
        self.losses = []

    def update_dataset(self, new_data):
        self.dataset.update_dataset(new_data)


    def training_step(self, epochs = 50):
        print('Dataset size:', self.dataset.pc_data.size())
        epoch = 0

        #sdf_tensor_dataset = TensorDataset(torch.tensor(self.dataset.pc_data, dtype=self.dtype), torch.tensor(self.dataset.sdf_data, dtype = self.dtype))
        #sdf_dataloader = DataLoader(sdf_tensor_dataset, batch_size = self.batch_size, shuffle = True)
        
        batch_points = self.dataset.pc_data
        batch_sdf = self.dataset.sdf_data
        batch_points.requires_grad_()

        while epoch < epochs:
            #epoch_loss = []
            #for batch_points, batch_sdf in sdf_dataloader:

            #batch_points.requires_grad_()
        
            self.optimizer.zero_grad()

            # Compute SDF preds and targets
            sdf_preds = self.model(batch_points)
            target_sdf_preds = batch_sdf

            # Compute grads
            sdf_gradient_preds = torch.autograd.grad(outputs=sdf_preds,
                                                        inputs=batch_points,
                                                        grad_outputs=torch.ones_like(sdf_preds, requires_grad=False, device=sdf_preds.device),
                                                        create_graph=True,
                                                        retain_graph=True,
                                                        only_inputs=True
                                                        )[0]
            
            # Compute loss
            self.total_sdf_loss = sdf_loss(sdf_preds, target_sdf_preds)
            self.total_eikonal_loss = eikonal_loss(sdf_gradient_preds)
            self.total_loss = self.sdf_loss_weight * torch.mean(self.total_sdf_loss) + self.eikonal_loss_weight * torch.mean(self.total_eikonal_loss)
            self.total_loss.backward()

            # Update weights
            self.optimizer.step()

            # Append loss
            self.losses.append(self.total_sdf_loss.mean().detach().cpu().numpy())
            #epoch_loss.append(self.total_loss)
            #print('Training epoch ',epoch + 1,'/',epochs,'|| Loss:',torch.tensor(epoch_loss).mean())
            print(f"Epoch: {epoch+1}/{epochs}, mean SDF loss: {torch.mean(self.total_sdf_loss)}, mean Eikonal loss: {torch.mean(self.total_eikonal_loss)}")

            epoch = epoch + 1

        return self.losses[-1]

# Initialize trainer

global_sdf_trainer = SDFTrainer(
    learning_rate= 4e-4,
    weight_decay= 0.012,
    sdf_loss_weight= 5.0,
    eikonal_loss_weight= 2.0,
    device= 'cuda'
)


# ======== MAIN MESSAGE HANDLER AND TRAINING LOOP ========


# Definition of Voxfield callback
def esdf_callback(esdf_layer_msg):
    global esdf_layer_msg_v
    esdf_layer_msg_v = esdf_layer_msg
    return

def PC_POS_callback_2topics(PC_msg, POS_msg):
    global x_pos, y_pos, z_pos, Q
    for transform in POS_msg.transforms:
        if transform.child_frame_id == "camera": 
            position = transform.transform.translation
            rotation = transform.transform.rotation
            x_pos = position.x
            y_pos = position.y
            z_pos = position.z
            q0 = rotation.w
            q1 = rotation.x
            q2 = rotation.y
            q3 = rotation.z
            Q = np.array([q0, q1, q2, q3])
    #print("PC MESSAGE TIME: ",PC_msg.header.stamp)
    #print("POS MESSAGE TIME: ", POS_msg.header.stamp)
    SIREN_Trainer(PC_msg)





def SIREN_Trainer(PC_msg):
    #----------------Parameters------------------------------------------------------------------
    dist_between_iter = 0.1 # Meters between two training iterations
    lidar_lim_max = 10 # Radius of the circle, centered in the drone, where LiDAR points are taken into account (meters)
    lidar_lim_min = 0.3
    num_epochs = 10 # NN hyperparameter
    batch_size = 64 # NN hyperparameter
    lambda_SDF = 5 # SDF loss weight
    lambda_eikonal = 2 # Eikonal loss weight
    learning_rate = 4e-4
    weight_decay = 0.012
    #num_val_points = 100 # Validation points to be considered between epochs

    global x_pos, y_pos, z_pos, x_last, y_last, z_last, Q, Q_last, total_wall_point_list, training_points, training_positions, global_sdf_trainer, iters

    acos_arg = np.abs(Q[0]*Q_last[0]+Q[1]*Q_last[1]+Q[2]*Q_last[2]+Q[3]*Q_last[3])
    if(acos_arg > 1):
        acos_arg = 1
    elif(acos_arg < -1):
        acos_arg = -1
    drone_pos = np.array([x_pos, y_pos, z_pos])
    if training_positions.shape[0] == 0:
        dist_to_closest_tp = dist_between_iter + 1 # Asures that the training is made if the list is empty
    else:
        tpkdtree = cKDTree(training_positions)
        dist_to_closest_tp, _ = tpkdtree.query(drone_pos) # Checks distance to the closest training position
    if(dist_to_closest_tp > dist_between_iter):
        training_positions = np.append(training_positions, [drone_pos], axis=0) # Add this position to the list of positions where training was performed
        R = RotQuad(Q) #Matriz de rotación del dron
        x_last = x_pos
        y_last = y_pos
        z_last = z_pos
        Q_last = Q
        pointcount = 0 #Contador de puntos detectados por el LiDAR
        wall_coordinates = np.empty((0,3)) #Para guardar los puntos con SDF=0 (paredes de objetos)
        for p in pc2.read_points(PC_msg, field_names = ("x", "y", "z"), skip_nans=True):
            #Move to global coordinates
            if((p[0]**2 + p[1]**2 + p[2]**2) < lidar_lim_max**2 and (p[0]**2 + p[1]**2 + p[2]**2) > lidar_lim_min**2): # Si está dentro del radio deseado, lo guarda como puntos de la pared
                plocal = np.array([p[0], p[1], p[2]])
                pglobal = drone_pos + R @ plocal
                #pglobal = plocal
                wall_coordinates =np.append(wall_coordinates, [pglobal], axis=0)
                total_wall_point_list = np.append(total_wall_point_list, [pglobal], axis=0)
                pointcount = pointcount + 1
        print("pointcount =", pointcount)
        print("wall_coordinates =", wall_coordinates)
        
        # BAGS SANTI
        #print("Total wall points: ", total_wall_point_list.shape[0])
        #np.save('lidar_points.npy', total_wall_point_list)
        #return
        
        #--PRINT LIDAR POINTS-- #
        #if (f_print_lidar_points == 1):
        #    fig_walls = go.Figure(data=[go.Scatter3d(x=wall_coordinates[:,0], y=wall_coordinates[:,1], z=wall_coordinates[:,2], mode='markers', marker=dict(size=5))])
        #    fig_walls.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='manual', aspectratio=dict(x=1, y=1, z=1)))
        #    fig_walls.show()

        training_points = np.empty((0,4)) # Reset the training points

        # ----------------Actualizar Voxfield con los puntos nuevos y obtener samples para el entrenamiento-----------------------
        
        global esdf_layer_msg_v, voxel_size, voxel_per_side, kEpsilon, kNumDataPacketsPerVoxel, block_size, meshgrid, num_voxfield_samples_surface, num_voxfield_samples_freespace
        global device, dtype
        # Get number of blocks and voxels in the message
        indices = np.asarray(esdf_layer_msg_v.indices).reshape(-1, 3)
        num_blocks = indices.shape[0]
        num_voxels = num_blocks * voxel_per_side**3

        # Initialize SDF data: array where each row is (x,y,z,SDF)
        voxfield_sdf_data = np.zeros((num_voxels, 4))
    
        # Find origin of each block
        origin = indices * block_size

        # Get SDF data and convert to float
        sdf_data_unit = esdf_layer_msg_v.data[::kNumDataPacketsPerVoxel]
        flag_data_unit = esdf_layer_msg_v.data[1::kNumDataPacketsPerVoxel]
        sdf = list(struct.unpack("<%df" % len(sdf_data_unit), struct.pack("<%dI" % len(sdf_data_unit), *sdf_data_unit)))
        observed = np.bitwise_and(flag_data_unit, True).astype(bool)

        # Find coordinates of voxels
        voxel_centers = np.expand_dims(meshgrid, 0) + np.expand_dims(origin, 1)
        voxel_centers = voxel_centers.reshape(-1, 3)
        voxfield_sdf_data[:, :3] = voxel_centers

        # Assign SDF values
        voxfield_sdf_data[:, 3] = sdf
        
        # Use only the observed data
        voxfield_sdf_data = voxfield_sdf_data[observed]

        # Find points close to the surface
        close_to_surface = (abs(voxfield_sdf_data[:, 3]) < 0.05).astype(bool)
        sdf_data_surface = voxfield_sdf_data[close_to_surface, :]
        sdf_data_freespace = voxfield_sdf_data[~close_to_surface, :]

        # Downsample if necessary
        if sdf_data_surface.shape[0] > num_voxfield_samples_surface:
            idx = np.random.choice(sdf_data_surface.shape[0], size = num_voxfield_samples_surface, replace = False)
            sdf_data_surface = sdf_data_surface[idx, :]
        sdf_data_surface = torch.from_numpy(sdf_data_surface).to(dtype = dtype, device = device)
        if sdf_data_freespace.shape[0] > num_voxfield_samples_freespace:
            idx = np.random.choice(sdf_data_freespace.shape[0], size = num_voxfield_samples_freespace, replace = False)
            sdf_data_freespace = sdf_data_freespace[idx, :]
        sdf_data_freespace = torch.from_numpy(sdf_data_freespace).to(dtype = dtype, device = device)
        voxfield_sdf_data_downsampled = torch.cat((sdf_data_surface, sdf_data_freespace), dim = 0)

        training_points = np.append(training_points, voxfield_sdf_data_downsampled, axis=0)

        # Update extents if necessary
        #extents[0][0] = min(extents[0][0], torch.min(voxfield_sdf_data_downsampled[:,0]))
        #extents[0][1] = min(extents[0][1], torch.min(voxfield_sdf_data_downsampled[:,1]))
        #extents[0][2] = min(extents[0][2], torch.min(voxfield_sdf_data_downsampled[:,2]))
        #extents[1][0] = max(extents[1][0], torch.max(voxfield_sdf_data_downsampled[:,0]))
        #extents[1][1] = max(extents[1][1], torch.max(voxfield_sdf_data_downsampled[:,1]))
        #extents[1][2] = max(extents[1][2], torch.max(voxfield_sdf_data_downsampled[:,2]))


        # ----------------Obtener puntos locales a través de la estimación del SDF por fuerza bruta-------------------------------
        truncation_dist = 0.2 # max distance to consider local points
        S = 1000 # rays to be considered
        Q = 20 # points per ray to be considered

        #Create the kdTree for the next step
        pkdtree = cKDTree(total_wall_point_list)

        if (pointcount < S):
            S = pointcount
        rnd_ray = np.random.choice(range(0, pointcount), S, replace=False) # Takes random samples of available rays
        for k in rnd_ray: #For each ray
            wall_point = np.array([wall_coordinates[k][0], wall_coordinates[k][1], wall_coordinates[k][2]]) # This is the wall point of that ray
            for l in range(Q): #For each point per ray
                dist_to_wall = random.uniform(-truncation_dist, truncation_dist)
                void_point = wall_point + ((wall_point-drone_pos)/np.linalg.norm(wall_point-drone_pos))*dist_to_wall # Coge puntos aleatorios a lo largo del rayo
                p_sdf_estimado, _ = pkdtree.query(void_point)
                if dist_to_wall > 0: # Si el punto está dentro de la pared, se cambia el signo del sdfestimado
                    p_sdf_estimado = -p_sdf_estimado
                new_tp = np.array([void_point[0], void_point[1], void_point[2], p_sdf_estimado])
                if truncation_dist >= np.abs(p_sdf_estimado): # If within bounds, add to the training point list
                    training_points = np.append(training_points,[new_tp], axis=0)
        
        # --PLOT TRAINING POINTS-- #
        if (f_training_points == 1):
            # Extracting coordinates and values
            x_tp_plt = training_points[:, 0]
            y_tp_plt = training_points[:, 1]
            z_tp_plt = training_points[:, 2]
            sdf_tp_plt = training_points[:, 3]

            fig = plt.figure()
            ax_tp_plt = fig.add_subplot(111, projection='3d')

            scatter_tp = ax_tp_plt.scatter(x_tp_plt, y_tp_plt, z_tp_plt, c=sdf_tp_plt, cmap='viridis', s=3)
            cbar_tp = plt.colorbar(scatter_tp)
            cbar_tp.set_label('Point Value')

            # Setting labels and title
            ax_tp_plt.set_xlabel('X')
            ax_tp_plt.set_ylabel('Y')
            ax_tp_plt.set_zlabel('Z')
            ax_tp_plt.set_title('3D Visualization of training points')

            plt.show()

        # --PLOT TRAINING POINTS BELOW CERTAIN VALUE-- #
        if (f_certain_training_points == 1):
            certain_training_points = np.empty((0,4))
            for row in training_points:
                if(np.abs(row[3]) <= tp_print_limit):
                    certain_training_points = np.append(certain_training_points,[row], axis=0)

            # Extracting coordinates and values
            x_tpc_plt = certain_training_points[:, 0]
            y_tpc_plt = certain_training_points[:, 1]
            z_tpc_plt = certain_training_points[:, 2]
            sdf_tpc_plt = certain_training_points[:, 3]

            fig = plt.figure()
            ax_tpc_plt = fig.add_subplot(111, projection='3d')

            scatter_tpc = ax_tpc_plt.scatter(x_tpc_plt, y_tpc_plt, z_tpc_plt, c=sdf_tpc_plt, cmap='viridis')
            cbar_tpc = plt.colorbar(scatter_tpc)
            cbar_tpc.set_label('Point Value')

            # Setting labels and title
            ax_tpc_plt.set_xlabel('X')
            ax_tpc_plt.set_ylabel('Y')
            ax_tpc_plt.set_zlabel('Z')
            ax_tpc_plt.set_title('3D Visualization of certain training points')

            plt.show()

        # --PLOT DISTRIBUCIÓN DE VALORES SDF DE LOS PUNTOS DE ENTRENAMIENTO
        if (f_sdf_tp_distribution == 1):
            distribution_objective = voxel_grid.flatten()
            plt.hist(distribution_objective, bins=70, color='blue', alpha=0.7)  # Adjust bins as needed
            plt.title('Distribution of SDF Values')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.show()

        # ----------------Entrenamiento de la SIREN--------------------------------------------------------------------------------
            
        # Training params
        epochs_warmstart = 50
        epochs_nominal = 10
        iter_warmstart = 5
            
        # Update dataset of SDF trainer
        training_points_torch = torch.from_numpy(training_points).to(dtype = torch.float, device='cuda')
        print('Training points torch', training_points_torch)
        global_sdf_trainer.update_dataset(training_points_torch)

        # Determine epochs
        current_epochs = epochs_warmstart if (iters < iter_warmstart) else epochs_nominal
        iters = iters + 1

        # Train
        _ = global_sdf_trainer.training_step(epochs = current_epochs)











        #criterion = CustomLoss(lambda_SDF, lambda_eikonal)
        #optimizer = optim.Adam(siren_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Load dataset (X_input, y_output) and move them to the GPU (if able). Then convert to float32 (expected by the NN)
        #X_input_batch = torch.tensor(training_points[:, :3])
        #y_output_batch = torch.tensor(training_points[:,3])
        #print(X_input_batch)
        #print(y_output_batch)
        #X_input_batch = X_input_batch.to(device)
        #y_output_batch = y_output_batch.to(device)
        #X_input_batch = X_input_batch.to(torch.float32)
        #y_output_batch = y_output_batch.to(torch.float32)
        #print("X_input_batch dtype:", X_input_batch.dtype)
        #print("y_output_batch dtype:", y_output_batch.dtype)

        # Create the DataLoader
        #point_dataset = TensorDataset(X_input_batch, y_output_batch)
        #train_loader = DataLoader(point_dataset, batch_size=batch_size, shuffle=True)


        # Train the model
        #for epoch in range(num_epochs):
            #running_loss = 0.0
            #val_mse_total = 0.0


            #for batch in train_loader:
                #inputs, targets = batch
                #targets = targets.view(-1,1)

                # Zero the parameter gradients
                #optimizer.zero_grad()

                # Forward pass
                #outputs = siren_model(inputs)
            
                # Compute gradients of the outputs w.r.t. the inputs
                #inputs.requires_grad = True
                #grad_output = torch.autograd.grad(outputs=siren_model(inputs), inputs=inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]
                #grad_output = torch.autograd.grad(outputs=siren_model(inputs), 
                                                  #inputs=inputs, 
                                                  #grad_outputs=torch.ones_like(outputs, requires_grad=False, device=outputs.device), 
                                                  #create_graph=True,
                                                  #retain_graph=True,
                                                  #only_inputs=True)[0]

                #print(" Gradiente hecho ")
                #loss = criterion(outputs, targets, grad_output)
                #loss.backward()
                #optimizer.step()

                #running_loss += loss.item()
            
            # Check validation each epoch
            #for k in range(num_val_points):
            #    val_input_tensor = torch.tensor([[val_point_list[k][0], val_point_list[k][1], val_point_list[k][2]]], dtype=torch.float32)
            #    val_output = NeRF(val_input_tensor.to(device))
            #    val_output_item = val_output.item()
            #    val_mse_total = val_mse_total + (val_output_item-val_pont_kdtree_sdf[k][0])**2

            #val_mse_total = val_mse_total/num_val_points

            #print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(train_loader)} ValLoss: {val_mse_total}')
            #print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(train_loader)}')

        # Save the trained model if needed
        #torch.save(siren_model.state_dict(), 'siren_model.pth')
        #print("Saved SIREN model")



# ======== MAIN FUNCTION / NODE ESTABLISHMENT ========
        
def main():
# Initialize the ROS node
    rospy.init_node('nerf_sdf', anonymous=True)
    rospy.Subscriber('/voxblox_node/esdf_map_out', DiscreteSDF,  esdf_callback, queue_size = 1, buff_size = 2**24)

    #Input topic selection
    topic_selector = 0


    if topic_selector == 0:
        topic_name_PC = "/velodyne_points"
        topic_name_POS = "/tf"
        PC_sub = message_filters.Subscriber(topic_name_PC, PointCloud2)
        POS_sub = message_filters.Subscriber(topic_name_POS, TFMessage)
        ts = message_filters.ApproximateTimeSynchronizer([PC_sub, POS_sub], queue_size=100000000, slop=0.1)
        ts.registerCallback(PC_POS_callback_2topics)
        print("Expecting /velodyne_points and /tf")

    elif topic_selector == 1:
        topic_name_PC = "/os1_cloud_node1/points"
        topic_name_POS = "/leica/pose/relative"
        PC_sub = message_filters.Subscriber(topic_name_PC, PointCloud2)
        POS_sub = message_filters.Subscriber(topic_name_POS, PoseStamped)
        ts = message_filters.ApproximateTimeSynchronizer([PC_sub, POS_sub], queue_size=100000000, slop=0.1)
        ts.registerCallback(PC_POS_callback_2topics)
        print("Expecting /os1_cloud_node1/points and /leica/pose/relative")
    else:
        print("Error: topic_selector value is not supported")

    # Keep the script running
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
