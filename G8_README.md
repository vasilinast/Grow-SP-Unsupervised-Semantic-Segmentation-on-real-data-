# GrowSP: Unsupervised Semantic Segmentation of Construction Site Point Clouds

This repository provides an implementation of unsupervised semantic segmentation for 3D construction site point clouds, adapted from the GrowSP approach. The main goal is to segment real-world point clouds without manual annotations, enabling efficient downstream tasks such as change detection and automated labeling support.

## Installation

### Docker Installation (Recommended)

We strongly recommend using docker, as it ensures a consistent, reproducible environment across systems and avoids compatibility issues commonly encountered with Conda installations (due to varying OS, GPU, and software dependencies).

To get started:

1. **Pull the Docker Image**

   ```bash
   docker pull yarinpour/growsp_nvidia:latest
   ```

2. **Install Required Python Libraries**

   Once inside the container, execute the following commands to adjust the Python dependencies:

   ```bash
   conda uninstall qhull
   conda uninstall pclpy
   conda install -c conda-forge/label/gcc7 qhull
   conda install -y -c conda-forge -c davidcaron pclpy
   ```

   Then, install the laspy package with its additional dependencies:

   ```bash
   pip install laspy[lazrs,laszip]
   ```

## Running the Code

A subset of our dataset is available for testing and demonstration purposes: [Download Dataset](#)

### 1. Pre-processing

Prepare your dataset for training by running the pre-processing script. This command tiles each point cloud, removes ground points, and generates initial regions based on our method. The results are organized into two subdirectories under your specified output path:

- `input_superpoints`: Contains the initial regions for each point cloud.
- `input_plys`: Contains the processed point clouds ready for training.

Run the following command:

```bash
python data_prepare/pre_processing_full_pipeline.py --input_path <path to raw point clouds> --output_path <path where to save the outputs>
```

A preprocessed dataset is also available here: [Preprocessed Dataset](#)

### 2. Training

Train the segmentation model with the following command. You can adjust hyperparameters as needed:

```bash
python train_const_site_yarin.py --data_path <path to input_plys folder> --sp_path <path to input_superpoints folder> --save_path <path to save the model> --pseudo_label_path <path to save pseudo labels> --voxel_size 0.15 --primitive_num 300 --batch_size 8 --growsp_start 80 --growsp_end 30
```

The training process will generate the output model, evaluation scores (such as silhouette score and within-cluster sum of squares), and a log file in the directory specified by `--save_path`.

Our trained models can be found here: [Trained Models](#)

### 3. Visual Evaluation and Inference

For visual evaluation and inference on new point cloud data, run the following command. Make sure that the `primitive_num` and `voxel_size` parameters match those used during training. You can set the number of segmentation classes using the `semantic_class` argument:

```bash
python post_processing_const_site.py --data_path <path to input point clouds in ply format> --save_path <path to the folder of the trained model> --vis_path <path to save the segmented point clouds> --semantic_class 18 --primitive_num 300 --voxel_size 0.3
```

For your reference, all segmented point clouds used for visual evaluation are stored on our HPC system at `/scratch/yarinp/vis`. The folder is divided into five subdirectories corresponding to different models:

- `model300`
- `model500`
- `model_grow_11500_2850`
- `model_grow_2875_712`
- `model_noground_v15`

Each model folder contains further subdirectories for point cloud segmentations with varying numbers of classes.
