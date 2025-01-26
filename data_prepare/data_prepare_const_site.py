import laspy
import MinkowskiEngine as ME
from os.path import join, exists, dirname, abspath
import numpy as np
import pandas as pd
import os, sys, glob
import argparse
from tile import tile

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from lib.helper_ply import write_ply

# Define script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,default='raw_data_const_site/samples_2', help='Path to raw .laz files')
parser.add_argument('--processed_data_path', type=str, default='data/construct_site/input', help='Path to save processed .ply files')
parser.add_argument('--sub_grid_size', type=float, default=0.010, help='Sub-grid size for downsampling')
args = parser.parse_args()

# Ensure the output directory exists
if not exists(args.processed_data_path):
    os.makedirs(args.processed_data_path)

def process_laz_file(laz_file, output_file, sub_grid_size):
    """
    Process a single .laz file into a downsampled .ply file.
    """
    # Open the .laz file with laspy
    with laspy.open(laz_file) as las_reader:
        las = las_reader.read()  # Read the file contents
        coords = np.vstack((las.x, las.y, las.z)).T
        colors = np.vstack((las.red, las.green, las.blue)).T.astype(np.uint8)

    # Downsample using MinkowskiEngine
    _, _, inds = ME.utils.sparse_quantize(
        np.ascontiguousarray(coords),
        features=colors,
        return_index=True,
        quantization_size=sub_grid_size
    )
    sub_coords = coords[inds]
    sub_colors = colors[inds]

    # Write the downsampled point cloud to a PLY file
    print(f"sub_coords shape: {sub_coords.shape}, sub_colors shape: {sub_colors.shape}")

    write_ply(output_file, [sub_coords, sub_colors], ['x', 'y', 'z', 'red', 'green', 'blue'])

# Main processing loop
print("Starting processing...")

print("tiling...")
tile_dir = join(args.data_path, "tiled")

tile(input_folder=args.data_path, output_dir=tile_dir, size="55x58", crop_min_y=18)

laz_files = [join(tile_dir, f) for f in os.listdir(tile_dir) if f.endswith('.laz') or f.endswith('.las')]
print(laz_files)

for laz_file in laz_files:
    print(f"Processing: {laz_file}")
    file_name = os.path.splitext(os.path.basename(laz_file))[0] + '.ply'  # Change file extension
    output_file = join(args.processed_data_path, file_name)
    process_laz_file(laz_file, output_file, args.sub_grid_size)

print("Processing completed.")
