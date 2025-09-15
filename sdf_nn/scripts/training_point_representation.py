#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped
from numpy import random
from scipy.spatial import cKDTree
import message_filters
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

#Complete point list (burrada)
total_wall_point_list = np.empty((0,3))

#Lista de puntos donde se han tomado muestras
training_positions = np.empty((0,3))

#Puntos de recuerdo
cont_lidar=0
saved_points = np.empty((0,4))

#Puntos de la pared escogidos (para posterior validación)
taken_wall_points = np.empty((0,4))


#Initialize variables
Q = np.array([1, 0, 0, 0])
Q_last = np.array([1, 0, 0, 0])
x_pos = 0
y_pos = 0
z_pos = 0
x_last = -1000
y_last = -1000
z_last = -1000


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


def PC_POS_callback_2topics(PC_msg, POS_msg):
    print("Synchronized message")
    #print("POS Seq: ", POS_msg.header.seq)
    #print("POS Stamp: ", POS_msg.header.stamp)
    #print("PC Seq: ", PC_msg.header.seq)
    #print("PC Stamp: ", PC_msg.header.stamp)
    global x_pos, y_pos, z_pos, Q
    x_pos = POS_msg.pose.position.x
    y_pos = POS_msg.pose.position.y
    z_pos = POS_msg.pose.position.z
    q0 = POS_msg.pose.orientation.w #Pendiente de revisar si las coordenadas son en este orden
    q1 = POS_msg.pose.orientation.x
    q2 = POS_msg.pose.orientation.y
    q3 = POS_msg.pose.orientation.z
    Q = np.array([q0, q1, q2, q3])
    NeRF_Trainer(PC_msg)


def NeRF_Trainer(PC_msg):
    #--Parameters--
    dist_between_iter = 1 # Meters between two training iterations
    #ang_between_iter = 20 # Degrees between two training iterations
    lidar_lim_max = 25 # Radius of the circle, centered in the drone, where LiDAR points are taken into account (meters)
    lidar_lim_min = 0.3
    max_rays = 500  # Max number of rays to be considered
    max_wall_points = 500 # Max number of wall points to be considered for training
    points_per_ray = 5 # Points per LiDAR ray
    learning_rate = 0.002 # NN hyperparameter
    num_epochs = 5 # NN hyperparameter
    batch_size = 64 # NN hyperparameter
    num_val_points = 100 # Validation points to be considered between epochs

    global x_pos, y_pos, z_pos, x_last, y_last, z_last, Q, Q_last, total_wall_point_list, saved_points, cont_lidar, taken_wall_points, training_positions
    #print(f"q0 = {Q[0]}, q1 = {Q[1]}, q2 = {Q[2]}, q3 = {Q[3]} || q0last = {Q_last[0]}, q1last = {Q_last[1]}, q2last = {Q_last[2]}, q3last = {Q_last[3]}")
    acos_arg = np.abs(Q[0]*Q_last[0]+Q[1]*Q_last[1]+Q[2]*Q_last[2]+Q[3]*Q_last[3])
    if(acos_arg > 1):
        acos_arg = 1
    elif(acos_arg < -1):
        acos_arg = -1
    #print(f"Argumento del arcocoseno: {asin_arg}")
    drone_pos = np.array([x_pos, y_pos, z_pos])
    if training_positions.shape[0] == 0:
        dist_to_closest_tp = dist_between_iter + 1 # Asures that the training is made if the list is empty
    else:
        tpkdtree = cKDTree(training_positions)
        dist_to_closest_tp, _ = tpkdtree.query(drone_pos) # Checks distance to the closest training position
    if(dist_to_closest_tp > dist_between_iter):
    #if (np.sqrt((x_pos-x_last)**2 + (y_pos-y_last)**2 + (z_pos-z_last)**2) > dist_between_iter or 180/np.pi*np.arccos(acos_arg) > ang_between_iter):
        training_positions = np.append(training_positions, [drone_pos], axis=0) # Add this position to the list of positions where training was performed
        print("x =", x_pos)
        print("y =", y_pos)
        print("z =", z_pos)
        dist_dif = np.sqrt((x_pos-x_last)**2 + (y_pos-y_last)**2 + (z_pos-z_last)**2)
        ang_dif = 180/np.pi*np.arccos(acos_arg)
        print("distance dif =", dist_dif)
        print("angle dif =", ang_dif)
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
                wall_coordinates =np.append(wall_coordinates, [pglobal], axis=0)
                total_wall_point_list = np.append(total_wall_point_list, [pglobal], axis=0)
                #print(" x : %f  y: %f  z: %f" %(pglobal[0],pglobal[1],pglobal[2]))
                pointcount = pointcount + 1
        print("pointcount =", pointcount)

        #------------------Choose random wall points for the training------------------
        num_samples_wallpoints = np.min([pointcount, max_wall_points]) #Wall points to be trained with
        rnd_samples1 = np.random.choice(range(0, pointcount), num_samples_wallpoints, replace=False)
        for k in rnd_samples1:
            new_tp = np.array([wall_coordinates[k][0], wall_coordinates[k][1], wall_coordinates[k][2], 0])
            saved_points = np.append(saved_points,[new_tp], axis=0)
            #training_points = np.append(training_points, [new_tp], axis=0)
            taken_wall_points = np.append(taken_wall_points, [new_tp], axis=0)

        #Create the kdTree for the next step
        pkdtree = cKDTree(total_wall_point_list)

        #--------Choose and estimate the sdf of points outside of the wall for training--------
        num_samp_ray = np.min([pointcount // 2, max_rays]) #LiDAR rays to be considered
        rnd_ray = np.random.choice(range(0, pointcount), num_samp_ray, replace=False) # Takes random samples of available rays
        print("rnd ray index = ",rnd_ray)
        for k in rnd_ray: #For each ray
            wall_point = np.array([wall_coordinates[k][0], wall_coordinates[k][1], wall_coordinates[k][2]]) #This is the wall point of that ray
            for l in range(1,points_per_ray + 1): #For each point per ray
                void_point = drone_pos + (wall_point-drone_pos)*random.triangular(0, 0.5, 1) # Coge puntos aleatorios a lo largo del rayo
                p_sdf_estimado, _ = pkdtree.query(void_point)
                new_tp = np.array([void_point[0], void_point[1], void_point[2], p_sdf_estimado])
                print(new_tp)
                saved_points = np.append(saved_points,[new_tp], axis=0)

        print("Shape of the saved_points array:", saved_points.shape)
        np.savetxt('saved_points_array.txt', saved_points)
        print('File saved')
        #training_points = np.append(training_points,saved_points,axis=0) # Add the general points
        if(x_pos > 0):
            # Create a 3D plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')


            # Scatter plot the points
            #ax.scatter(saved_points[:, 0], saved_points[:, 1], saved_points[:, 2], c='b', marker='o')
            ax.scatter(taken_wall_points[:, 0], taken_wall_points[:, 1], taken_wall_points[:, 2], c='b', marker='o')

            # Set axis labels
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')

            # Show the plot
            plt.show()
        


def main():
    # Initialize the ROS node
    rospy.init_node('training_point_representation', anonymous=True)

    #Input topic selection
    topic_selector = 1


    if topic_selector == 0:
        topic_name_PC = "/velodyne_points"
        topic_name_POS = "/ground_truth_to_tf/pose"
        PC_sub = message_filters.Subscriber(topic_name_PC, PointCloud2)
        POS_sub = message_filters.Subscriber(topic_name_POS, PoseStamped)
        ts = message_filters.ApproximateTimeSynchronizer([PC_sub, POS_sub], queue_size=1000000, slop=0.1)
        ts.registerCallback(PC_POS_callback_2topics)

    elif topic_selector == 1:
        topic_name_PC = "/os1_cloud_node1/points"
        topic_name_POS = "/leica/pose/relative"
        PC_sub = message_filters.Subscriber(topic_name_PC, PointCloud2)
        POS_sub = message_filters.Subscriber(topic_name_POS, PoseStamped)
        ts = message_filters.ApproximateTimeSynchronizer([PC_sub, POS_sub], queue_size=10000000, slop=0.1)
        ts.registerCallback(PC_POS_callback_2topics)
    else:
        print("Error: topic_selector value is not supported")

    # Keep the script running
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass