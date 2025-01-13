import open3d as o3d
import numpy as np
from os.path import join, exists, dirname, abspath
import sys, glob

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from lib.helper_ply import read_ply as read_ply


def inspect_ply_file(file_path):
    """
    Reads a PLY file and prints the header and first two data points.

    Args:
        file_path (str): Path to the PLY file.
    """
    try:
        # Read the PLY file
        ply_data = o3d.io.read_point_cloud(file_path)
        
        # Get the points from the PLY file
        points = ply_data.points
        if len(points) == 0:
            print("No point data found in the PLY file.")
            return
        
        # Print the header (Open3D doesn't directly expose headers, so approximated)
        print("=== HEADER ===")
        with open(file_path, 'r') as file:
            for line in file:
                print(line.strip())
                if line.strip() == "end_header":
                    break

        # Print the first two points
        print("\n=== FIRST TWO DATA POINTS ===")
        print("Point 1:", points[0])
        if len(points) > 1:
            print("Point 2:", points[1])
        else:
            print("Only one point available in the file.")
        
        # Check if other data (e.g., colors, normals) are available
        if len(ply_data.colors) > 0:
            print("\n=== COLOR INFORMATION ===")
            print("Color 1:", ply_data.colors[0])
            if len(ply_data.colors) > 1:
                print("Color 2:", ply_data.colors[1])
        
        if len(ply_data.normals) > 0:
            print("\n=== NORMAL INFORMATION ===")
            print("Normal 1:", ply_data.normals[0])
            if len(ply_data.normals) > 1:
                print("Normal 2:", ply_data.normals[1])

    except Exception as e:
        print(f"Error reading PLY file: {e}")




def inspect_npy_file(file_path):
    """
    Inspects the columns, dimensions, and structure of a .npy file.
    
    Args:
        file_path (str): Path to the .npy file.
    """
    try:
        # Load the .npy file
        data = np.load(file_path)
        
        # Print basic information
        print(f"=== INSPECTION OF '{file_path}' ===")
        print(f"Data Type: {type(data)}")
        print(f"Shape: {data.shape}")
        print(f"Data Type of Elements: {data.dtype}")
        print(data[0])
        print(np.unique(data))
        
        # Check if the data is structured (e.g., a structured array or ndarray)
        if isinstance(data, np.ndarray):
            if data.dtype.names:  # Check for structured array with named columns
                print("\nColumns (Structured Array):", data.dtype.names)
                print("\nSample Row:")
                print(data[0])  # Print the first row
            else:
                print("\nSample Data (First 5 Elements):")
                print(data[:5])  # Print the first 5 elements of the array
        
        else:
            print("\nUnsupported data structure.")
    
    except Exception as e:
        print(f"Error inspecting file: {e}")


# Provide the path to your .ply file
file_path = "/workspace/data/S3DIS/input/Area_1_copyRoom_1.ply"
data = read_ply(file_path)
print(data.shape)
print(data[0])

# Provide the path to your .npy file
file_path = "/workspace/data/S3DIS/initial_superpoints/Area_1_copyRoom_1_superpoint.npy"
inspect_npy_file(file_path)