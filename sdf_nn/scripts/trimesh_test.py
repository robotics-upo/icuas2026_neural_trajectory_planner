import numpy as np

def calculate_sdf(voxel_coords, data_points):
    # Function to calculate SDF for a single voxel
    # Implement your SDF calculation method here
    return np.min(np.linalg.norm(voxel_coords - data_points, axis=1))

def update_sdf_map(voxel_grid, data_points):
    # Function to update the SDF map for the entire grid
    for idx, voxel_coords in np.ndenumerate(voxel_grid):
        sdf_value = calculate_sdf(voxel_coords, data_points)
        voxel_grid[idx] = sdf_value

# Example usage
voxel_size = (20, 20, 20)
voxel_grid = np.full(voxel_size)
data_points = np.array([(3.4, 5.6, 7.8), (6.7, 8.9, 2.3)])  # Example data points

update_sdf_map(voxel_grid, data_points)

# Accessing the updated SDF map
print(voxel_grid)