import numpy as np
from scipy import stats
import os
from os.path import join, exists, dirname, abspath
import sys, glob
import h5py

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from lib.helper_ply import read_ply, write_ply
import time
import MinkowskiEngine as ME
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
colormap = []
for _ in range(1000):
    for k in range(12):
        colormap.append(plt.cm.Set3(k))
    for k in range(9):
        colormap.append(plt.cm.Set1(k))
    for k in range(8):
        colormap.append(plt.cm.Set2(k))
colormap.append((0, 0, 0, 0))
colormap = np.array(colormap)
import argparse
ignore_label = 12
voxel_size = 0.05
vis = True

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='data/S3DIS/input', help='raw data path')
args = parser.parse_args()


def voxelize(path):
    f = Path(path)
    data = read_ply(f)
    coords = np.vstack((data['x'], data['y'], data['z'])).T.copy()
    feats = np.vstack((data['red'], data['green'], data['blue'])).T.copy()
    labels = data['class'].copy()
    coords = coords.astype(np.float32)
    coords -= coords.mean(0)

    '''Voxelize'''
    scale = 1 / voxel_size
    coords = np.floor(coords * scale)
    print(f"voxalizing {path}...")
    coords, feats, labels, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords),
                            feats, labels=labels, ignore_label=-1, return_index=True, return_inverse=True)
    
    save_directory = f.parent  # Same directory as the original .ply file
    save_name = f.stem + ".h5"  # Use the same base name with .h5 extension
    output_file = save_directory / save_name

    # Save the voxelization output using h5py
    with h5py.File(output_file, "w") as f:
        f.create_dataset("coords", data=coords)
        f.create_dataset("feats", data=feats)
        f.create_dataset("labels", data=labels)
        f.create_dataset("unique_map", data=unique_map)
        f.create_dataset("inverse_map", data=inverse_map)

    print(f"Voxelization data saved to {output_file}")

if __name__ == "__main__":
    voxelize(args.input_path)

