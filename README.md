<div align="center">

# SceneHub  
**A Dataset and Evaluation Framework for 6-DoF 4D Scenes**

<img src="assets/teaser_figure.png" width="100%"/>

### [📄 Paper](XX) | [🌐 Project Page](https://scenehub4d.github.io/) | [📂 Dataset](https://www.dropbox.com/scl/fo/gfskqntptl6vemn4d62jb/...) | [🎮 Dataset Viewer](https://scenehub4d.github.io/#scene-viewers)

</div>

## Contents

- [Dataset Access](#dataset-access)  
- [Dataset Structure](#dataset-structure)  
- [1. Setup](#1-setup)  
  - [1.1 Environment Setup](#11-environment-setup)  
  - [1.2 Build Instructions](#12-build-instructions)  
- [2. 4D Scene Reconstruction](#2-4d-scene-reconstruction)  
  - [2.1 Point Cloud or Textured Mesh](#21-point-cloud-or-textured-mesh)  
  - [2.2 Gaussian Splat](#22-gaussian-splat)  
    - [2.2.1 Training a Gaussian Splat](#221-training-a-gaussian-splat)  
    - [2.2.2 Rendering a Gaussian Splat](#222-rendering-a-gaussian-splat)  
- [3. Evaluation](#3-evaluation)  
  - [3.1 2D SI/TI Measurement](#31-2d-siti-measurement)  
  - [3.2 Volumetric TI Measurement](#32-volumetric-ti-measurement)  
  - [3.3 Geometry Complexity Score (GCS) Measurement](#33-geometry-complexity-score-gcs-measurement)  
  - [3.4 Multi-view Rendering using Camera Pose Variants](#34-multi-view-rendering-using-camera-pose-variants)  
  - [3.5 Visualize and Generate Camera Poses](#35-visualize-and-generate-camera-poses)  
  - [3.6 Visualize Dynamic Point Clouds](#36-visualize-dynamic-point-clouds)

## Dataset Access

A partial version of the SceneHub dataset is available for public download:

**📦 Download link**: [SceneHubData](https://www.dropbox.com/scl/fo/gfskqntptl6vemn4d62jb/ACKZ8XfLVs8YA_EOushYDoM?rlkey=wj7engjmfmefwtl9nql5plf23&st=p0zilhf7&dl=0)

To ensure ease of access, we provide:
- **100 RGB-D frames** per scene (instead of the full 1000+)
- **Example geometry outputs** (1 frame per scene) including Gaussian Splatting, mesh, and point cloud

This partial set is sufficient for prototyping and evaluating pipeline compatibility.

⚠️ The **full dataset**, including all RGB-D frames, will be made available soon once long-term hosting is arranged. Stay tuned!


## Dataset Structure

<details>
<summary>📁 Click to expand dataset structure</summary>

<br>


📌 **Note**: 
Scene labels differ slightly from the paper for clarity.(`lab area → arena`, `factory → mill19`).  
The directory structure here differs slightly from the original layout presented in the paper, for structural simplicity:


```bash
MM_release/
├── camera_pose/                # camera pose metadata
├── geometry/                   # 3D reconstruction outputs
├── photogrammetry/             # high-res background meshes
├── rgbd_data/                  # raw RGB-D frames
└── dataset_layout.txt          # detailed layout of the dataset structure
```

### RGBD data
```bash
dataset-paper/rgbd_data/
└── <scene>/                      # scene ∈ {arena, couch, kitchen, mill19, whiteboard}
    └── <scene>_scene<id>_100/    # id ∈ {0,1,2,…}
        └── rgbd/
            └── cam<cam_id>/      # cam_id ∈ {0,1,2,3}
                ├── rgb/          # color frames (.png)
                └── trans_depth/  # aligned depth maps (.png)
```

### Camera Pose
```bash
camera_pose/
├── cam_views/                    # per-scene view files ∈ {original, shifted, interpolated, random}
│       └── <scene>/              # scene ∈ {arena, couch, kitchen, mill19, whiteboard}
│               ├── <scene>_original.npy
│               ├── <scene>_shifted.npy
│               ├── <scene>_interpolated.npy
│               └── <scene>_random_extrinsics_seed42_sample50.npy
├── extrinsics/                   # camera extrinsic matrices (world to camera)
│       └── extrinsics_<scene>.npy                     
└── intrinsics/                   # camera factory intrinsics, optimal intrinsics parameters
    └── <scene>_config.json                     
```

### Geometry Output
```bash
geometry/
└── <scene>/                          # scene ∈ {arena, couch, kitchen, mill19, 
        └── <scene>_scene<id>_100/    # id ∈ {0,1,2,…}
            ├── 3dgs/                 # Gaussian splatting outputs
            │   └── point_cloud/
            │       └── iteration_30000/point_cloud_<frame_idx>.ply 
            ├── mesh/                 # textured meshes
            │   ├── <frame_idx>.obj
            │   ├── <frame_idx>.mtl
            │   └── <frame_idx>.png        
            └── ptcl/                 # raw point clouds
                 <frame_idx>.pcl
```

### Photogrammetry
```bash
photogrammetry/
└── <scene>/                      # scene-specific photogrammetry
    ├── <scene>.obj               # mesh file
    ├── <scene>.mtl               # mesh file
    └── <scene>.png               # mesh file
```
</details>




## 1. Setup

### 1.1 Environment Setup
To install the required dependencies, run the following setup script:

```bash
bash setup.sh
conda activate 3d
```

### 1.2 Build Instructions

After setting up the environment, build the source code as follows:

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

## 2. 4D Scene Reconstruction 

### 2.1 Point Cloud or Textured Mesh

Choose your output type via the `--reconstruct_geometry` flag:

```bash
python reconstruct.py \
    --rgbd_path ${DATA_DIR} \
    --frame_count ${FRAME_COUNT} \
    --output_dir ${OUTPUT_DIR} \
    --intrinsic_path ${INTRINSIC_PATH} \
    --extrinsic_path ${EXTRINSIC_PATH} \
    --reconstruct_geometry <color_ptcl | textured_mesh> 
```
Alternatively, to generate all geometry types for all scenes in the dataset, simply run:
```
bash reconstruct.sh
```

### 2.2 Gaussian Splat

#### 2.2.1. Training a Gaussian Splat 

This pipeline builds on the official [3D Gaussian Splatting codebase](https://github.com/graphdeco-inria/gaussian-splatting), but instead of relying on SfM-based sparse point clouds from RGB images, we use point clouds derived from depth images.

---
Use `gaussian_splatting_rgbd/train.py` to train a scene-specific Gaussian Splat representation.

```bash
python train.py \
    -s ${DATA_PATH} \                          # Path to input RGB-D
    --eval \                                   # Enable evaluation during training
    --model_path ${MODEL_PATH_SAVED} \         # Directory to save model checkpoints
    --save_iterations 7000 30000 \             # Save model at these iterations
    --resolution 1 \                           # Resolution scale (1 = original resolution)
    --test_iterations 1000 \                   # Run evaluation every 1000 iterations
    --intrinsic_path ${INTRINSIC_PATH} \       # Path to intrinsics JSON file
    --extrinsic_path ${EXTRINSIC_PATH} \       # Path to extrinsics NPY file
    --ptcl_downsample 0 \                      # Voxel size for point cloud downsampling (0 = no downsampling)
                                               # Increase to >0 (e.g., 0.01–0.05) for sparser point clouds
    --frame_idx ${FRAME_IDX}                   # Temporal frame index to train on (e.g., 0 = first frame)
```
Or simply use the following script (edit hardcoded paths inside the script as needed):
```bash
bash run_train.sh # for a single scene
bash train_multiple_scenes.sh # for multiple scenes
```

#### 2.2.2. Rendering a Gaussian Splat 

```bash
python render.py \
    --iteration 30000 \                  # Checkpoint iteration to render
    --model_path ${MODEL_PATH_SAVED}     # Path to the saved model directory
```

## 3. Evaluation

### 3.1 2D SI/TI Measurement

Measuring Spatial Information (SI) and Temporal Information (TI) from RGB and depth frames for a multi-camera RGB-D scene: 

```bash
python measure_si_ti.py \
    --scene_dir ${SCENE_DIR}/ \           # Path to scene folder (e.g., arena/arena_scene0_100)
    --output_dir ${LOG_OUTPUT_DIR} \      # Directory to save log output
    --max_frames 100    
```

### 3.2 Volumetric TI Measurement

Computing Volumetric Temporal Information (V-TI) across time using 3D point clouds: 
```bash
python measure_3d_si_ti.py \
    --scene_dir ${DATA_SOURCE}/${DATA_NAME}/gt/color_ptcl \  # Directory with point cloud frames (.ply)
    --scene_name ${DATA_NAME} \                              # Scene identifier for output
    --max_frames ${MAX_FRAMES} \                             # Number of frames to evaluate
    --output_dir ../ours_logs \                              # Output directory for logs
    --unit m 
```

### 3.3 Geometry Complexity Score (GCS) Measurement

Analyzes structural mesh complexity across a sequence.
```bash
python measure_mesh_complexity_sequence.py \
    --mesh_dir ${DATA_SOURCE}/${DATA_NAME}/gt/mesh \     # Directory containing mesh files (.obj or .ply)
    --scene_name ${DATA_NAME} \                          # Scene name for labeling results
    --max_frames ${MAX_FRAMES} \                         # Number of mesh frames to process
    --output_dir ../ours_logs  
```

### 3.4 Multi-view Rendering using Camera Pose Variants

This pipeline renders 3D scenes (mesh or point cloud) from multiple camera views using provided intrinsics and extrinsics. It also supports novel view generation via pose interpolation, shifting, and random sampling.

---

```bash
python multiview_renderer.py \
    --cam_idx_list "0 1 2 3" \                            # Cameras to render
    --data_path ${DATA_DIR} \                            # Input data root (must contain geometry)
    --mode gt \                                          # Rendering mode: 'gt' for ground truth geometry
    --frame_count ${FRAME_COUNT} \                       # Frame index (e.g., 1)
    --output_dir ${OUTPUT_DIR} \                         # Where to save rendered images
    --reconstruct_geometry textured_mesh \               # Geometry type (e.g., textured_mesh or color_ptcl)
    --extrinsic_path ${EXTRINSIC_PATH} \                 # Global extrinsic path (.npy)
    --intrinsic_path ${INTRINSIC_PATH} \                 # Camera intrinsics path (.json)
    --scene_name kitchen                                 # Scene name label
```

### 3.5 Visualize and Generate Camera Poses

This script visualizes the 3D mesh of a scene along with various camera poses (original, interpolated, shifted, random). It also saves the generated camera extrinsics for later use.

```bash
python visualize_camera_poses.py \
    --scene arena \
    --scene_name scene1 \
    --add_background  # optional, shows an additional background mesh
```

### 3.6 Visualize Dynamic Point Clouds
Visualizing a sequence of .ply point cloud files as a dynamic animation using Open3D. 
```bash
python visualize_pcd.py --data_path /path/to/pointclouds
```