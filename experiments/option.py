from argparse import ArgumentParser


parser = ArgumentParser()           
parser.add_argument('--rgbd_path', type=str, default="../output/")
parser.add_argument('--frame_count', type=int, required=True)
parser.add_argument('--test_idx', type=int, default=None)
parser.add_argument('--output_dir', type=str, required=True)

parser.add_argument('--intrinsic_path', type=str, required=True)
parser.add_argument('--extrinsic_path', type=str, required=True)

parser.add_argument('--cam_idx_list', type=int, nargs='+', default=[0, 1, 2, 3],
                    help="List of camera indices (e.g., --cam_idx_list 0 1 2 3)")

parser.add_argument('--max_depth', type=int, default=8000)
parser.add_argument('--min_depth', type=int, default=0)

parser.add_argument('--reconstruct_geometry', type=str, choices=['color_ptcl', 'ptcl', 'textured_mesh', 'mesh'], 
                    default='ptcl', help="Choose the final reconstructed representation")
parser.add_argument('--mesh_voxel_size', type=float, default=0.01, 
                    help="Voxel size for mesh representation. 1cm by default.")
parser.add_argument('--ptcl_downsample_voxel_size', type=float, default=0.0, 
                    help="Voxel size for point cloud representation. If 0.0, raw point cloud will be used.")
parser.add_argument('--decimation_ratio', type=float, default=0.0, 
                    help="Choose the final reconstructed representation")

args=parser.parse_args()

if "arena" in args.rgbd_path:
    args.scene_type = "arena"
elif "couch" in args.rgbd_path:
    args.scene_type = "couch"
elif "kitchen" in args.rgbd_path:
    args.scene_type = "kitchen"
elif "whiteboard" in args.rgbd_path:
    args.scene_type = "whiteboard"
elif "mill19" in args.rgbd_path:
    args.scene_type = "mill19"
else:
    raise ValueError("Invalid scene type in rgbd_path. Please specify a valid scene type (arena, couch, kitchen, whiteboard, mill19).")