#!/bin/bash
#SBATCH --job-name=docker_to_singularity
#SBATCH --output=train_test-%j.out
#SBATCH --error=train_test-%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --export=ALL,SINGULARITY_CACHDIR=/tmp/drive,SINGULARITY_TMPDIR=/tmp/drive

DATASET=ConstSite_dummy2
BATCHSIZE=4
VOXELSIZE=0.3

# Load Singularity module
module load singularity/4.0.2
echo "loaded singularity"
echo $PATH

# Set working directory and container path
WORK_PATH=~/htcv_ss2425_dlfor3d

DATA_PATH=/scratch/${DATASET}
SAVE_PATH==/scratch/results/${DATASET}_bs${BATCHSIZE}_vs${VOXELSIZE}
CONTAINER_PATH=~/growsp_nvidia_latest.sif

# Create required folders if needed
# mkdir -p ${WORK_PATH}/data/saved_models

# Execute the Singularity container with GPU support
singularity exec --nv \
  --bind ${WORK_PATH}:/workspace \
  --bind /scratch/juliadora:/scratch \
  --pwd /workspace \
  ${CONTAINER_PATH} \
  bash -c "python train_const_site_yarin.py \
    --data_path ${DATA_PATH}/training_data_ply \
    --sp_path  ${DATA_PATH}/initial_superpoints/training_data_ply \
    --save_path ${SAVE_PATH} \
    --batch_size ${BATCHSIZE} \
    --voxel_size ${VOXELSIZE}"




