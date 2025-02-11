import open3d as o3d
import numpy as np
from scipy import stats
from os.path import join, exists, dirname, abspath
import sys, glob
from sklearn.cluster import DBSCAN
from tile import tile
import laspy
import MinkowskiEngine as ME
import CSF

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from lib.helper_ply import read_ply, write_ply
import time
import os
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

colormap = []
for _ in range(10000):
    for k in range(12):
        colormap.append(plt.cm.Set3(k))
    for k in range(9):
        colormap.append(plt.cm.Set1(k))
    for k in range(8):
        colormap.append(plt.cm.Set2(k))
colormap.append((0, 0, 0, 0))
colormap = np.array(colormap)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='raw_data_const_site/preprocess_pipeline_test/raw', help='raw data path')
parser.add_argument('--output_path', type=str, default='raw_data_const_site/preprocess_pipeline_test/outputs', help='Path to save outputs.')
parser.add_argument('--sub_grid_size', type=float, default=0.010, help='Sub-grid size for downsampling')
parser.add_argument('--sp_thin', type=float, default=0.30, help='Sub-grid size for downsampling large SPs.')
parser.add_argument('--vis', type=bool, default=True, help='Create vis folder of initial SPs.')

args = parser.parse_args()


# Ensure the output directory exists
if not exists(args.output_path):
    os.makedirs(args.output_path)


def ransac(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=1000)
    return np.array(inliers)

def thin_road_points(coords, road_index, voxel_size=args.sp_thin):
    # Extract road points
    road_coords = coords[road_index]

    # Perform sparse quantization with MinkowskiEngine
    _, inds = ME.utils.sparse_quantize(
        np.ascontiguousarray(road_coords),
        return_index=True,
        quantization_size=voxel_size
    )

    # Get downsampled points and their original indices
    thinned_road_index = road_index[inds]

    return np.array(thinned_road_index, dtype=int)

def construct_superpoints(coords, rgbs):
    #f = Path(path)
    #data = read_ply(f)
    #coords = np.vstack((data['x'], data['y'], data['z'])).T.copy()
    #rgbs = np.vstack((data['red'], data['green'], data['blue'])).T.copy()

    #labels = data['class'].copy()
    #labels -= 1
    coords = coords.astype(np.float32)
    coords -= coords.mean(0)

    rgbs = rgbs.astype(np.float32)
    #rgbs -= rgbs.mean(0)

    '''RANSAC'''
    print("Starting ransac..")
    road_index = ransac(coords)
    road_mask = np.zeros(coords.shape[0], dtype=bool)
    road_mask[list(road_index)] = True  # Mark road indices as True
    other_index = np.where(~road_mask)[0]

    #other_index = []
    #for i in range(coords.shape[0]):
    #    if i not in road_index:
    #        other_index.append(i)

    # Spatial DBSCAN (initial clustering based on spatial coordinates)    
    print("Starting dbscan")
    #other_index = np.array(other_index)
    other_coords = coords[other_index]  # *self.voxel_size
    other_rgbs = rgbs[other_index]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(other_coords)
    #pcd.points = o3d.utility.Vector3dVector(other_rgbs)
    other_region_idx = np.array(pcd.cluster_dbscan(eps=0.2, min_points=10))
    
    # Assign labels to the points based on DBSCAN results
    sp_labels = -np.ones(coords.shape[0], dtype=np.int32)
    sp_labels[other_index] = other_region_idx
    road_label = other_region_idx.max() + 1
    sp_labels[road_index] = road_label
    
    
    # Perform second DBSCAN for each spatial cluster again
    unique_labels = np.unique(sp_labels[(sp_labels != -1) & (sp_labels != road_label)])
    all_rgb_sub_clusters = []
    
    for label in unique_labels:
        cluster_mask = sp_labels == label
        cluster_coords = coords[cluster_mask]
        cluster_rgbs = rgbs[cluster_mask]

        # Compute pairwise RGB distances (Euclidean)
        # RGB values should be normalized to [0, 1] before calculating distance if needed.
        #distance_matrix = np.linalg.norm(cluster_rgbs[:, np.newaxis] - cluster_rgbs, axis=-1)
        
        # Perform DBSCAN with smaller eps value
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(cluster_coords)
        rgb_sub_labels = np.array(pcd2.cluster_dbscan(eps=0.05, min_points=10))
        #rgb_sub_labels = DBSCAN(eps=0.1, min_samples=5).fit(cluster_rgbs)
        # Assign new labels for the RGB sub-clusters
        sp_labels[cluster_mask] = rgb_sub_labels + sp_labels.max() + 1  # Unique labels for each sub-cluster

    return sp_labels, road_index

def process_laz_file(laz_file, file_name, sub_grid_size):
    """
    Process a single .laz file into a downsampled .ply file.
    """
    # Open the .laz file with laspy
    with laspy.open(laz_file) as las_reader:
        las = las_reader.read()  # Read the file contents
        coords = np.vstack((las.x, las.y, las.z)).T
        colors = np.vstack((las.red, las.green, las.blue)).T.astype(np.uint8)

        classifications = las.classification

    # Downsample using MinkowskiEngine
    _, _, inds = ME.utils.sparse_quantize(
        np.ascontiguousarray(coords),
        features=colors,
        return_index=True,
        quantization_size=sub_grid_size
    )
    sub_coords = coords[inds]
    sub_colors = colors[inds]
    sub_classifications = classifications[inds]

    sp_labels, road_index = construct_superpoints(sub_coords, sub_colors)

    print("Thinning plane superpoints...")
    thinned_road_index = thin_road_points(sub_coords, road_index)
    # Create a mask to keep only the remaining points
    keep_mask = np.ones(sub_coords.shape[0], dtype=bool)
    keep_mask[road_index] = False  # First mark all road points for removal
    keep_mask[thinned_road_index] = True  # Keep only the thinned road points

    # Apply filtering
    coords = sub_coords[keep_mask]
    rgbs = sub_colors[keep_mask]
    sp_labels = sp_labels[keep_mask]  # Ensure labels remain consistent
    classifications = sub_classifications[keep_mask]

    non_ground_mask = classifications != 2  # Keep only non-ground points
    coords = coords[non_ground_mask]
    rgbs = rgbs[non_ground_mask]
    sp_labels = sp_labels[non_ground_mask]

    sp_output_folder = join(args.output_path, "input_superpoints")
    ply_output_folder = join(args.output_path, "input_plys")

    if not exists(sp_output_folder):
        os.makedirs(sp_output_folder)
    if not exists(ply_output_folder):
        os.makedirs(ply_output_folder)

    print("Saving files...")
    # Write the downsampled point cloud to a PLY file
    output_file = join(ply_output_folder, file_name)
    write_ply(output_file, [coords, rgbs], ['x', 'y', 'z', 'red', 'green', 'blue'])

    #Saving super points
    np.save(join(sp_output_folder, file_name[:-4]+'_superpoint.npy'), sp_labels)

    if args.vis:
        print("starting visualization..")
        vis_path = join(args.output_path, 'vis')
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)
        colors = np.zeros_like(coords)
        for p in range(colors.shape[0]):
            colors[p] = 255 * (colormap[sp_labels[p].astype(np.int32)])[:3]
        colors = colors.astype(np.uint8)

        write_ply(vis_path + '/' + file_name, [coords, colors], ['x', 'y', 'z', 'red', 'green', 'blue'])

    print('completed scene: {}'.format(file_name))

def csf(laz_file, classified_dir):
    inFile = laspy.read(laz_file) # read a las file
    points = inFile.points
    xyz = np.vstack((inFile.x, inFile.y, inFile.z)).transpose() # extract x, y, z and put into a list

    csf = CSF.CSF()

    # prameter settings
    csf.params.bSloopSmooth = True
    csf.params.cloth_resolution =1
    csf.params.rigidness = 5
    csf.params.time_step = 0.45
    csf.params.class_threshold = 0.1
    csf.params.interations = 100

    # more details about parameter: http://ramm.bnu.edu.cn/projects/CSF/download/

    csf.setPointCloud(xyz)
    ground = CSF.VecInt()  # a list to indicate the index of ground points after calculation
    non_ground = CSF.VecInt() # a list to indicate the index of non-ground points after calculation
    csf.do_filtering(ground, non_ground, exportCloth=False) # do actual filtering.

    # Assign classification values: ground (2), non-ground (1)
    classification = np.zeros(len(inFile.points), dtype=np.uint8)
    classification[np.array(ground)] = 2
    classification[np.array(non_ground)] = 1

    # Update classification field
    inFile.classification = classification

    # Get the input file name
    original_filename = os.path.basename(laz_file)

    # Define the output folder in the same directory as the input file
    os.makedirs(classified_dir, exist_ok=True)

    # Define the output file path
    output_file = os.path.join(classified_dir, original_filename)

    # Write the classified LAZ file
    inFile.write(output_file)

    print(f"Classified LAZ file saved at: {output_file}")


    return

# Main processing loop
print("Starting processing...")
print("Classifying...")
classified_dir = join(args.output_path, "ground_classified")
laz_files = [join(args.input_path, f) for f in os.listdir(args.input_path) if f.endswith('.laz') or f.endswith('.las')]

for laz_file in laz_files:
    print(f"Classifying: {laz_file}")
    csf(laz_file, classified_dir)

print("tiling...")
tile_dir = join(args.output_path, "tiled")
tile(input_folder=classified_dir, output_dir=tile_dir, size="55x58", crop_min_y=18)
laz_files = [join(tile_dir, f) for f in os.listdir(tile_dir) if f.endswith('.laz') or f.endswith('.las')]
print(laz_files)

for laz_file in laz_files:
    print(f"Processing: {laz_file}")
    file_name = os.path.splitext(os.path.basename(laz_file))[0] + '.ply'  # Change file extension
    output_file = join(args.output_path, file_name)
    process_laz_file(laz_file, file_name, args.sub_grid_size)