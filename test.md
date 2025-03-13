# Unsupervised Semantic Segmentation of Construction Site Point Clouds

This repository provides an implementation of unsupervised semantic segmentation for 3D construction site point clouds, adapted from the GrowSP approach. The main goal is to segment real-world point clouds without manual annotations, enabling efficient downstream tasks such as change detection and automated labeling support.

## Installation

### docker Installation (Recommended)

We strongly recommend using docker, as it ensures a consistent, reproducible environment across systems and avoids compatibility issues commonly encountered with Conda installations (due to varying OS, GPU, and software dependencies).

To get started:

1. Clone the repository:
   ```bash
   git clone https://github.com/tub-cv-group/htcv_ss2425_dlfor3d
   cd htcv_ss2425_dlfor3d
