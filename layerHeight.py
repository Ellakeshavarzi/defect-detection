import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_point_cloud(file_path):
    # Load the point cloud file (PLY format)
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def preprocess_point_cloud(pcd):
    # Remove outliers and noise
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd

def extract_layer_heights(pcd):
    # Convert point cloud to numpy array
    points = np.asarray(pcd.points)
    
    # Filter based on height (Y-axis)
    y_coords = points[:, 1]  # Y-axis represents height
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    # Bin the points by height to identify layers
    bin_size = 1.0  # Adjust bin size based on your brick's layer height in mm
    bins = np.arange(min_y, max_y, bin_size)

    # Group points into bins (representing layers)
    layer_heights = []
    for i in range(len(bins) - 1):
        layer_points = points[(y_coords >= bins[i]) & (y_coords < bins[i + 1])]
        if len(layer_points) > 50:  # Threshold to ignore noise
            avg_height = np.mean(layer_points[:, 1])
            layer_heights.append(avg_height)

    return np.array(layer_heights)

def plot_layer_heights(layer_heights):
    # Plot the calculated layer heights
    plt.figure(figsize=(8, 5))
    plt.plot(layer_heights, marker='o', linestyle='-', color='blue', label='Layer Heights')
    plt.xlabel("Layer Index")
    plt.ylabel("Height (mm)")
    plt.title("3D Printed Brick Layer Heights")
    plt.grid(True)
    plt.show()

# Load the point cloud (change the file path as needed)
file_path = "brick_point_cloud.ply"  # Replace with your actual file path
pcd = load_point_cloud(file_path)

# Preprocess the point cloud to remove noise
cleaned_pcd = preprocess_point_cloud(pcd)

# Extract layer heights
layer_heights = extract_layer_heights(cleaned_pcd)

# Print the layer heights
print("Detected Layer Heights (mm):")
print(layer_heights)

# Plot the layer heights
plot_layer_heights(layer_heights)

# Visualize the processed point cloud
o3d.visualization.draw_geometries([cleaned_pcd])
