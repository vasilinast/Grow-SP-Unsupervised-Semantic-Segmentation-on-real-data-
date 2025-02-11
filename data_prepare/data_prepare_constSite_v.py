import laspy
import numpy as np
import open3d as o3d
import MinkowskiEngine as ME
from sklearn.neighbors import BallTree
from os.path import join, exists, dirname, abspath
import os, sys, argparse
from tile import tile
from lib.helper_ply import write_ply

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

# Define script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='raw_data_const_site/samples_2', help='Path to raw .laz files')
parser.add_argument('--processed_data_path', type=str, default='data/construct_site/input', help='Path to save processed .ply files')
parser.add_argument('--radius', type=float, default=0.05, help='Radius for density estimation')
parser.add_argument('--base_keep_ratio', type=float, default=0.1, help='Minimum fraction of points to keep')
parser.add_argument('--sub_grid_size', type=float, default=0.010, help='Sub-grid size for sparse quantization')
parser.add_argument('--method', type=str, choices=['sparse_quantization', 'density_based'], default='sparse_quantization', help='Thinning method to use')
parser.add_argument('--skip_tiling', action='store_true', help='Skip the tiling step')
args = parser.parse_args()

# Ensure the output directory exists
if not exists(args.processed_data_path):
    os.makedirs(args.processed_data_path)

def load_laz_file(file_path):
    """Load a .laz file and return point coordinates and colors."""
    with laspy.open(file_path) as las_reader:
        las = las_reader.read()
        coords = np.vstack((las.x, las.y, las.z)).T
        colors = np.vstack((las.red, las.green, las.blue)).T.astype(np.uint8)
    return coords, colors

def compute_radial_density(coords, radius):
    """Compute local density using a fixed-radius search."""
    tree = BallTree(coords) #It's a data structure for nearest neighbor search. It organizes points in a multi-dimensional space into tree structure.
    densities = tree.query_radius(coords, r=radius, count_only=True)
    return densities

def adaptive_thinning(coords, colors, base_keep_ratio, radius):
    """Perform density-based adaptive thinning."""
    densities = compute_radial_density(coords, radius)
    max_density = np.max(densities) #For normalization --> in order for the probs to be from 0 to 1.
    keep_probs = base_keep_ratio + (1 - densities / max_density) * (1 - base_keep_ratio) #Computes the prob that each point should be kept.
    sampled_mask = np.random.rand(len(coords)) < keep_probs #To prevent deterministic behavior. For points with high prob(sparse), there is higher prob that the random number will be < keep_probs (which is high).
    return coords[sampled_mask], colors[sampled_mask]

def sparse_quantization(coords, colors, sub_grid_size):
    """Perform sparse quantization-based downsampling.""" #Uniform downsampling.
    _, _, inds = ME.utils.sparse_quantize(
        np.ascontiguousarray(coords),
        features=colors,
        return_index=True,
        quantization_size=sub_grid_size
    )
    return coords[inds], colors[inds]

def visualize_point_cloud(coords, title="Point Cloud Visualization"):
    """Visualize a point cloud using Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    o3d.visualization.draw_geometries([pcd], window_name=title)

def process_laz_file(laz_file, output_file, method, base_keep_ratio, radius, sub_grid_size):
    """Load, thin, visualize, and save the point cloud."""
    coords, colors = load_laz_file(laz_file)
    print(f"Original points: {coords.shape[0]}")
    #visualize_point_cloud(coords, title="Original Point Cloud")
    
    if method == 'density_based':
        downsampled_coords, downsampled_colors = adaptive_thinning(coords, colors, base_keep_ratio, radius)
    else:
        downsampled_coords, downsampled_colors = sparse_quantization(coords, colors, sub_grid_size)
    
    print(f"Downsampled points: {downsampled_coords.shape[0]}")
    #visualize_point_cloud(downsampled_coords, title="Thinned Point Cloud")
    
    write_ply(output_file, [downsampled_coords, downsampled_colors], ['x', 'y', 'z', 'red', 'green', 'blue'])

# Main processing loop
print("Starting processing...")

# Tiling step (conditionally executed)
if not args.skip_tiling:
    print("Tiling...")
    tile_dir = join(args.data_path, "tiled")
    tile(input_folder=args.data_path, output_dir=tile_dir, size="55x58", crop_min_y=18)
else:
    print("Skipping tiling step.")

# Process all laz files
laz_files = [join(tile_dir, f) for f in os.listdir(tile_dir) if f.endswith('.laz') or f.endswith('.las')]
print(laz_files)

for laz_file in laz_files:
    print(f"Processing: {laz_file}")
    file_name = os.path.splitext(os.path.basename(laz_file))[0] + '_thinned.ply' #changed file extension!
    output_file = join(args.processed_data_path, file_name)
    process_laz_file(laz_file, output_file, args.method, args.base_keep_ratio, args.radius, args.sub_grid_size)

print("Processing completed.")