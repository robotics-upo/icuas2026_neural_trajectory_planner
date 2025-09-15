#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import numpy as np
from sdf_nn.srv import sdfService

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from nav_msgs.msg import OccupancyGrid

#See if the nn can be placed in the GPU
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class NeuralNetwork(nn.Module): # Se crea la red haciendo subclassing del nn.Module (estructura igual al paper del iSDF)
    def __init__(self): # Se inicializan las capas en el __init__
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_softplus_stack = nn.Sequential(
            nn.Linear(3, 256),
            nn.Softplus(),
            nn.Linear(256, 256),
            nn.Softplus(),
            nn.Linear(256, 256),
            nn.Softplus(),
            nn.Linear(256, 256),
            nn.Softplus(),
            nn.Linear(256, 1)
        )

    def forward(self, x): # Se implementan las operaciones sobre los datos de entrada en el forward
        x = self.flatten(x)
        logits = self.linear_softplus_stack(x)
        return logits

NeRF = NeuralNetwork().to(device) # Instanciamos la red neuronal y la movemos a la GPU

# Construct the path to the pre-trained weights
file_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(file_dir, 'nn_weights')
pre_trained_weights_path = os.path.join(models_dir, 'nerf_model_santi_03.pth')  # Adjust the file name accordingly


NeRF.load_state_dict(torch.load(pre_trained_weights_path))
NeRF.eval()  # Set the model to evaluation mode

def handle_nerf_point_request(req):
    
    input_tensor = torch.tensor([[req.coordinates.x, req.coordinates.y, req.coordinates.z]], dtype=torch.float32)
    output = NeRF(input_tensor.to(device))
    output_item = output.item()

    return output_item

def add_two_ints_server():
    rospy.init_node('nerf_sdf_service')
    s = rospy.Service('nerf_sdf_request', sdfService, handle_nerf_point_request)
    print("NerF Point Request Service Ready")
    rospy.spin()

if __name__ == "__main__":
    add_two_ints_server()
