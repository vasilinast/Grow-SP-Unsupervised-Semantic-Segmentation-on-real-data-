#!/bin/bash
#SBATCH --job-name=docker_to_singularity
#SBATCH --output=container_test-%j.out
#SBATCH --error=container_test-%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2


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
  --pwd /workspace \
  ${CONTAINER_PATH} \
  bash -c "python3 text.py"


