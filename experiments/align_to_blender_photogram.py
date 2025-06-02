from rgbd_processor import RGBD_Processor
from rgbd_processor import blender_to_world_transform
import numpy as np

CAMERA_PARAMS = {
    "arena": {
        # "loc": (1.7334,  2.3829,  -1.2887),
        # "eul_deg": (420.6, 18.921, -54.566),
        "loc": (2.2548,  2.328,  -0.85997),
        "eul_deg": (58.749,  20.377,  -55.532),
    },
    "couch": {
        # "loc": (-0.853268, 1.89047, -1.6608),
        # "eul_deg": (132.697, -194.743, 50.1609),
        "loc": (-1.6673,  1.947,  -1.4323),
        "eul_deg": (297.02,  3.1026,  -133.77),     
    },
    "kitchen": {
        # "loc": (3.25123, 1.58253, 0.483055),
        # "eul_deg": (140.259, -375.62, -434.86),
        "loc": (2.2752,  1.5561,  -1.1269),
        "eul_deg": (-87.091,  -182.29,  -249.99),    
    },
    "whiteboard": {
        # "loc": (1.54115, 1.50426, -2.08984),
        # "eul_deg": (78.2128, -6.96374, -70.2771),
        "loc": (1.8795,  1.5185,  -1.8559),
        "eul_deg": (453.74,  5.3172,  288.21),        
    },
    "mill19": {
        "loc": (-1.74, 0.82, -0.18),
        "eul_deg": (90, -90, 1.438),
    },
}

def process_scene(scene_type: str,
                  cam_idx_list: list,
                  data_root: str = "/XX/XX"):
    params = CAMERA_PARAMS.get(scene_type)
    if params is None:
        raise ValueError(f"No camera params for scene '{scene_type}'")
    location   = params["loc"]
    euler_deg  = params["eul_deg"]

    data_path       = f"{data_root}/rgbd_data/{scene_type}/{scene_type}_scene1"
    intrinsic_path  = f"{data_root}/intrinsics/{scene_type}_config.json"
    extrinsic_path  = f"{data_root}/extrinsics/global_extrinsics_{scene_type}.npy"

    processor = RGBD_Processor(cam_idx_list=cam_idx_list,
                               extrinsic_path=extrinsic_path,
                               intrinsic_path=intrinsic_path)
    processor.load_images(data_path, frame_count=1, test_idx=1)

    extrinsics = blender_to_world_transform(
        relative_extrinsics=processor.extrinsics,
        location=location,
        euler_deg_xyz=euler_deg
    )

    out_static    = f"/XX/XX/extrinsics/global_extrinsics_{scene_type}_photogram_aligned.npy"
    out_scaniverse= f"/XX/XX/extrinsics/global_extrinsics_{scene_type}_photogram_aligned.npy"
    np.save(out_static,    extrinsics)
    np.save(out_scaniverse, extrinsics)

if __name__ == "__main__":
    SCENE      = "kitchen"
    CAM_INDICES= [0, 1, 2, 3]
    process_scene(SCENE, CAM_INDICES)