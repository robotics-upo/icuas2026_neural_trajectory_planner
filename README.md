Neural ESDF local trajectory planner implementation

This repository contains every package used in the framework presented for ICUAS2026 conference. It is an open-source framework that integrates a distance-aware 3D local continuous trajectory planning algorithm based on Euclidean Signed Distance Fields (ESDFs) with an online generated Sinusoidal Representation Neural network (SIREN) to estimate the required ESDF. 

# **Installation**

For installation purpouses, please follow the instructions in the source packages, found in the following URLs:

https://github.com/SamsungLabs/HIO-SDF/tree/main

https://github.com/robotics-upo/Heuristic_path_planners

https://github.com/RAFALAMAO/hector-quadrotor-noetic/tree/main

https://github.com/HKUST-Aerial-Robotics/FIESTA (for benchmarking porpouses)

Take into the account that the original property of the individual packages (except for Heuristic_path_planners package) does not belong to us in any way. The HIO package, Heuristic_path_planners and hector_quadrotor_noetic here are modified from the original versions. HIO-SDF system was slightly modified to provide an interface with our system, and some parameters of the training system and network were also changed. In the Heuristic_path_planners package the modifications are large and can be tracked in the original repository, in the "sdf_local" branch, where we created a local planning system from scratch. The hector_quadrotor_noetic package now includes all the files neccesary to replicate the experimental validation in our paper. Please follow the installation instructions on the original repositories and then change the files in the packages for the ones here.

# **How to cite**

The paper related to this framework, **"C-3TO: Continuous 3D Trajectory Optimization on Neural Euclidean Signed Distance Fields"** was presented to ICUAS2026. You can cite it with the following information:

Pending publication 

# **Acknowledgement**

![Logo Ministerio de Ciencia e Innovación, Fondos NextGenerationEU, PRTR, Agencia Estatal de Investigación](./acknowledgement_logos.png)

This work was supported by the grants COBUILD (PID2024-161069OB-C31, funded by the Ministry of Science, Innovation and Universities, the Spanish Research Agency, and the European Regional Development Fund, MICIU /AEI /10.13039/501100011033 / FEDER, UE), and PICRAH 4.0 (PLEC2023-010353, funded by the Spanish Research Agency and the Ministry of Science, Innovation and Universities, MCIN /AEI /10.13039/501100011033)
