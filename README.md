# SceneHub

## Dataset Access

A partial version of the SceneHub dataset is available for public download:

**📦 Download link**: [SceneHubData](https://bit.ly/SceneHubData)

To ensure ease of access, we provide:
- **100 RGB-D frames** per scene (instead of the full 1000+)
- **Example geometry outputs** (1 frame per scene) including Gaussian Splatting, mesh, and point cloud

This partial set is sufficient for prototyping and evaluating pipeline compatibility.

⚠️ The **full dataset**, including all RGB-D frames, will be made available soon once long-term hosting is arranged. Stay tuned!

---

## Datset structure
📌 **Note**: The directory structure here differs slightly from the original layout presented in the paper.  
To enhance accessibility and structural clarit, we have rearranged the dataset contents.
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
└── <scene>/                      # scene name
    └── <scene>_scene<id>_100/    # id ∈ {0,1,2,…}
        └── rgbd/
            └── cam<cam_id>/      # cam_id ∈ {0,1,2,3}
                ├── rgb/          # color frames (.png)
                └── trans_depth/  # aligned depth maps (.png)
```

### Camera Pose
```bash
camera_pose/
├── cam_views/                    # per-scene view files
│       └── <scene>/                # scene ∈ {arena, couch, kitchen, mill19, whiteboard}
│               ├── <scene>_interpolated.npy
│               ├── <scene>_original.npy
│               ├── <scene>_random_extrinsics_seed42_sample50.npy
│               └── <scene>_shifted.npy
│       └── couch/                 # view definitions (JSON)
├── extrinsics/                   # camera extrinsic matrices
│       └── extrinsics_<scene>.npy                     # 4×4 matrices in plain text                
└── intrinsics/                   # camera intrinsic parameters
    └── <scene>_config.json                     # focal lengths, principal points, etc.
```

### Geometry Output
```bash
geometry/
└── <scene>/                      # scene name
        └── <scene>_scene<id>_100/    # id ∈ {0,1,2,…}
            ├── 3dgs/                 # Gaussian splatting outputs
            │   └── point_cloud/
            │       └── iteration_30000/point_cloud_<frame_idx>.ply  # point cloud file
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
    ├── <scene>.obj                     # mesh file
    ├── <scene>.mtl                     # mesh file
    └── <scene>.png                     # mesh file
```
