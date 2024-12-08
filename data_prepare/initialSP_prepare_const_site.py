from pclpy import pcl
import pclpy
import numpy as np
from scipy import stats
import os
from os.path import join, exists, dirname, abspath
import sys, glob
import torch
from scipy.spatial import KDTree

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

# =================== Predefined Parameters ===================

ignore_label = 12
voxel_size = 0.2
vis = True

#VCCS
#https://pcl.readthedocs.io/projects/tutorials/en/latest/supervoxel_clustering.html
supervoxel_resolution = 0.2 #determines the leaf size of the underlying octree structure (in meters)
supervoxel_seed_resolution = 0.5 #determines seed resolution
spatial_importance = 0.6 #higher values will result in supervoxels with very regular shapes 
normal_importance = 1.5 #how much surface normals will influence the shape of the supervoxels
color_importance = 0.4 #how much color will influence the shape of the supervoxels

#Region growing
#https://pcl.readthedocs.io/projects/tutorials/en/master/region_growing_segmentation.html
region_growing_min_size = 10 ##after the segmentation is done all clusters that have less points than minimum(or have more than maximum) will be discarded. The default values for minimum and maximum are 1 and ‘as much as possible’ respectively.
region_growing_max_size = 10000000
region_growing_neighbors = 10
region_growing_smooth_threshold = 3.0 #* 180.0/np.pi
region_growing_curvature_threshold = 1
region_growing_residual_threshold = 1


# =================== Argument Parser ===================

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='data/construct_site/input/test', help='raw data path')
parser.add_argument('--sp_path', type=str, default='data/construct_site/initial_superpoints')
args = parser.parse_args()

# =================== Functions ===================

def supervoxel_clustering(coords, rgb=None):
    pc = pcl.PointCloud.PointXYZRGBA(coords, rgb)
    normals = pc.compute_normals(radius=3, num_threads=2)
    vox = pcl.segmentation.SupervoxelClustering.PointXYZRGBA(voxel_resolution=supervoxel_resolution, seed_resolution=supervoxel_seed_resolution)
    vox.setInputCloud(pc)
    vox.setNormalCloud(normals)
    vox.setSpatialImportance(spatial_importance)
    vox.setNormalImportance(normal_importance)
    vox.setColorImportance(color_importance)
    output = pcl.vectors.map_uint32t_PointXYZRGBA()
    vox.extract(output)
    return list(output.items())

def region_growing_simple(coords):
    pc = pcl.PointCloud.PointXYZ(coords)
    normals = pc.compute_normals(radius=3, num_threads=2)
    clusters = pclpy.region_growing(pc, normals=normals, min_size=region_growing_min_size, max_size=region_growing_max_size, n_neighbours=region_growing_neighbors,
                                    smooth_threshold=region_growing_smooth_threshold, curvature_threshold=region_growing_curvature_threshold, residual_threshold=region_growing_residual_threshold)
    return clusters, normals.normals


def construct_superpoints(path):
    f = Path(path)
    data = read_ply(f)
    coords = np.vstack((data['x'], data['y'], data['z'])).T.copy()
    feats = np.vstack((data['red'], data['green'], data['blue'])).T.copy()
    #labels = data['class'].copy()
    coords = coords.astype(np.float32)
    coords -= coords.mean(0)

    time_start = time.time()
    '''Voxelize'''
    print("voxelizing...")
    scale = 1 / voxel_size
    coords = np.floor(coords * scale)
    coords, feats, unique_map, inverse_map = ME.utils.sparse_quantize(
        np.ascontiguousarray(coords), feats, return_index=True, return_inverse=True
    )
    coords = coords.numpy().astype(np.float32)

    '''VCCS'''
    print("superpoint clustering...")
    out = supervoxel_clustering(coords, feats)
    voxel_idx = -np.ones(coords.shape[0], dtype=np.int32)

    voxel_num = 0
    print("iterating over SPs...")
    kdtree = KDTree(coords)
    print(len(out))
    for voxel in range(len(out)):
        if out[voxel][1].voxels_.xyz.shape[0] >= 0:
            for xyz_voxel in out[voxel][1].voxels_.xyz:
                #index_colum = np.where((xyz_voxel == coords).all(1))
                _, index_colum = kdtree.query(xyz_voxel)
                voxel_idx[index_colum] = voxel_num
            voxel_num += 1

    '''Region Growing'''
    print("region growing simple...")
    clusters = region_growing_simple(coords)[0]
    region_idx = -1 * np.ones(coords.shape[0], dtype=np.int32)
    for region in range(len(clusters)):
        for point_idx in clusters[region].indices:
            region_idx[point_idx] = region

    '''Merging'''
    print("merging...")
    merged = -np.ones(coords.shape[0], dtype=np.int32)
    voxel_idx[voxel_idx != -1] += len(clusters)
    for v in np.unique(voxel_idx):
        if v != -1:
            voxel_mask = v == voxel_idx
            voxel2region = region_idx[voxel_mask] ### count which regions are appeared in current voxel
            dominant_region = stats.mode(voxel2region)[0][0]
            if (dominant_region == voxel2region).sum() > voxel2region.shape[0] * 0.5:
                merged[voxel_mask] = dominant_region
            else:
                merged[voxel_mask] = v

    '''Make Superpoint Labels Continuous'''
    print("making SP labels continous...")
    sp_labels = -np.ones_like(merged)
    count_num = 0
    for m in np.unique(merged):
        if m != -1:
            sp_labels[merged == m] = count_num
            count_num += 1

    '''ReProject to Input Point Cloud'''
    print("reprojecting to point cloud...")
    out_sp_labels = sp_labels[inverse_map]
    out_coords = np.vstack((data['x'], data['y'], data['z'])).T
    #out_labels = data['class'].squeeze()
    #
    if not exists(args.sp_path):
        os.makedirs(args.sp_path)
    np.save(args.sp_path + '/' + f.name[:-4] + '_superpoint.npy', out_sp_labels)
    print("saved ")

    if vis:
        print("creating visualization...")
        vis_path = args.sp_path +'/vis/'
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)
        colors = np.zeros_like(out_coords)
        for p in range(colors.shape[0]):
            colors[p] = 255 * (colormap[out_sp_labels[p].astype(np.int32)])[:3]
        colors = colors.astype(np.uint8)
        write_ply(vis_path + '/' + f.name, [out_coords, colors], ['x', 'y', 'z', 'red', 'green', 'blue'])

    #sp2gt = -np.ones_like(out_labels)
    #for sp in np.unique(out_sp_labels):
    #    if sp != -1:
    #        sp_mask = sp == out_sp_labels
    #        sp2gt[sp_mask] = stats.mode(out_labels[sp_mask])[0][0]

    print('completed scene: {}, used time: {:.2f}s'.format(f.name, time.time() - time_start))
    return out_sp_labels



if torch.cuda.is_available():
    print("CUDA is available. Device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")

print('start constructing initial superpoints')
path_list = []
folders = sorted(glob.glob(args.input_path + '/*.ply'))
for _, file in enumerate(folders):
    path_list.append(file)
pool = ProcessPoolExecutor(max_workers=1)
result = list(pool.map(construct_superpoints, path_list))

print('end constructing initial superpoints')

all_labels, all_sp2gt = [], []
for (labels, sp2gt) in result:
    mask = (sp2gt != -1) & (sp2gt != ignore_label)
    labels, sp2gt = labels[mask].astype(np.int32), sp2gt[mask].astype(np.int32)
    all_labels.append(labels), all_sp2gt.append(sp2gt)

all_labels, all_sp2gt  = np.concatenate(all_labels), np.concatenate(all_sp2gt)
sem_num = 12
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
