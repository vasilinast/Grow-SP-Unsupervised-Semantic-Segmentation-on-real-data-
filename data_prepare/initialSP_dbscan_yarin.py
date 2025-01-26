import open3d as o3d
import numpy as np
from scipy import stats
from os.path import join, exists, dirname, abspath
import sys, glob
from sklearn.cluster import DBSCAN

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
parser.add_argument('--input_path', type=str, default='data/construct_site/input/test2', help='raw data path')
parser.add_argument('--sp_path', type=str, default='data/construct_site/initial_superpoints')
args = parser.parse_args()

vis = True


def ransac(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=1000)
    return np.array(inliers)


def construct_superpoints(path):
    f = Path(path)
    data = read_ply(f)
    coords = np.vstack((data['x'], data['y'], data['z'])).T.copy()
    rgbs = np.vstack((data['red'], data['green'], data['blue'])).T.copy()

    #labels = data['class'].copy()
    #labels -= 1
    coords = coords.astype(np.float32)
    coords -= coords.mean(0)

    rgbs = rgbs.astype(np.float32)
    #rgbs -= rgbs.mean(0)

    time_start = time.time()
    name = join(f.parts[-2], f.name)

    '''RANSAC'''
    print("starting ransac..")
    road_index = ransac(coords)
    road_mask = np.zeros(coords.shape[0], dtype=bool)
    road_mask[list(road_index)] = True  # Mark road indices as True
    other_index = np.where(~road_mask)[0]

    #other_index = []
    #for i in range(coords.shape[0]):
    #    if i not in road_index:
    #        other_index.append(i)

    # Spatial DBSCAN (initial clustering based on spatial coordinates)    
    print("starting dbscan")
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


    #Saving results
    if not os.path.exists(join(args.sp_path, f.parts[-2])):
        os.makedirs(join(args.sp_path, f.parts[-2]))
    np.save(join(args.sp_path, name[:-4]+'_superpoint.npy'), sp_labels)

    if vis:
        print("starting visualization..")
        vis_path = join(args.sp_path, 'vis', f.parts[-2])
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)
        colors = np.zeros_like(coords)
        for p in range(colors.shape[0]):
            colors[p] = 255 * (colormap[sp_labels[p].astype(np.int32)])[:3]
        colors = colors.astype(np.uint8)

        out_coords = np.vstack((data['x'], data['y'], data['z'])).T
        write_ply(vis_path + '/' + f.name, [out_coords, colors], ['x', 'y', 'z', 'red', 'green', 'blue'])

    #sp2gt = -np.ones_like(coords.shape[0])
    #for sp in np.unique(sp_labels):
    #    if sp != -1:
    #        sp_mask = sp == sp_labels
    #        sp2gt[sp_mask] = stats.mode(labels[sp_mask])[0][0]

    print('completed scene: {}, used time: {:.2f}s'.format(name, time.time() - time_start))
    #return (labels, sp2gt)


print('start constructing initial superpoints')
# Get a single list of all .ply files in the input directory
ply_files = np.sort(glob.glob(join(args.input_path, "*.ply")))

# Assign this list to be processed
path_list = list(ply_files)
print(path_list)

#pool = ProcessPoolExecutor(max_workers=1)
#result = list(pool.map(construct_superpoints, path_list))
for path in path_list:
    construct_superpoints(path)
#construct_superpoints(path_list[0])
print('end constructing initial superpoints')

''''
all_labels, all_sp2gt = [], []
for (labels, sp2gt) in result:
    mask = (sp2gt != -1)
    labels, sp2gt = labels[mask].astype(np.int32), sp2gt[mask].astype(np.int32)
    all_labels.append(labels), all_sp2gt.append(sp2gt)

all_labels, all_sp2gt  = np.concatenate(all_labels), np.concatenate(all_sp2gt)
sem_num = 19
mask = (all_labels >= 0) & (all_labels < sem_num)
histogram = np.bincount(sem_num * all_labels[mask] + all_sp2gt[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)
o_Acc = histogram[range(sem_num), range(sem_num)].sum() / histogram.sum()
tp = np.diag(histogram)
fp = np.sum(histogram, 0) - tp
fn = np.sum(histogram, 1) - tp
IoUs = tp / (tp + fp + fn + 1e-8)
m_IoU = np.nanmean(IoUs)
s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
for IoU in IoUs:
    s += '{:5.2f} '.format(100 * IoU)
print(' Acc: {:.5f}  Test IoU'.format(o_Acc), s)

#result = list(pool.map(construct_superpoints, test_path_list))
print(' Acc: {:.5f}  Test IoU'.format(o_Acc), s)
'''''