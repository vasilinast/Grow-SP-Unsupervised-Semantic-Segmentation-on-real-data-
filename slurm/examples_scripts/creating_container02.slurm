#!/bin/bash
#SBATCH --job-name=docker_to_singularity
#SBATCH --output=h_test-%j.out
#SBATCH --error=h_test-%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --export=ALL,SINGULARITY_CACHDIR=/tmp/drive,SINGULARITY_TMPDIR=/tmp/drive

# Load Singularity module
module load singularity/4.0.2
echo "loaded singularity"
echo $PATH

# Set working directory and container path
WORK_PATH=~/htcv_ss2425_dlfor3d
CONTAINER_PATH=~/growsp_nvidia_latest.sif

# Create required folders if needed
# mkdir -p ${WORK_PATH}/data/saved_models

# Execute the Singularity container with GPU support
singularity exec --nv \
  --bind ${WORK_PATH}:/workspace \
  --bind /scratch/juliadora:/scratch \
  --pwd /workspace \
  ${CONTAINER_PATH} \
  bash -c "pip  install lazrs && python3 data_prepare/data_prepare_const_site.py --data_path /scratch/ConstSite/training_data/  --processed_data_path /scratch/ConstSite/training_data_ply/"


