import os, cv2
import numpy as np
import open3d as o3d

from rgbd_processor import RGBD_Processor
from multiview_renderer import *

def reconstruct_mesh(args, rgbd_processor=None):    
    rgbd_processor.load_images(data_path=args.rgbd_path, test_idx=args.test_idx, frame_count=args.frame_count, load_memory=False)        

    if args.decimation_ratio > 0.0:
        save_path = f"{args.output_path}_decimated_{args.decimation_ratio}"
    else:
        save_path = args.output_path
    # rgbd_processor.rgbd_to_textured_mesh(save_path=save_path, decimation_ratio=args.decimation_ratio)        
    rgbd_processor.rgbd_to_textured_mesh_use_cpp(save_path=save_path, decimation_ratio=args.decimation_ratio)
    return rgbd_processor.geometry_paths

def reconstruct_ptcl(args, rgbd_processor=None):
    rgbd_processor.load_images(data_path=args.rgbd_path, test_idx=args.test_idx, frame_count=args.frame_count)

    if args.ptcl_downsample_voxel_size > 0.0:
        save_path = f"{args.output_path}_decimated_{args.decimation_ratio}"
    else:
        save_path = args.output_path

    rgbd_processor.rgbd_to_ptcl(save_path=save_path)
    return rgbd_processor.geometry_paths

def test_multiview_ssim(args):
    rgbd_processor = RGBD_Processor(cam_idx_list=args.cam_idx_list, extrinsic_path=args.extrinsic_path, intrinsic_path=args.intrinsic_path)
    intrinsics, _, _, extrinsics = rgbd_processor.get_calibration()    
    
    mesh_path = "/home/XX/workspace/3d-measurement/output_scaniverse_aligned/arena_scene1/gt/textured_mesh/0.obj"
    mesh_path = "/home/XX/workspace/3d-measurement/texture_mesh_test/arena_scene1.obj"

    # mesh = o3d.io.read_triangle_mesh("/home/XX/3d-measurement/output/gt/textured_mesh_gt/0.obj", True)
    
    # renderer = o3d.visualization.rendering.OffscreenRenderer(1920, 1080)    

    # renderer = MultiViewMeshRenderer(mesh_path)
    
    with MultiViewMeshRendererContext(mesh_path) as renderer:
        render_imgs = renderer.render_views(intrinsics, extrinsics, args.cam_idx_list)

        os.makedirs("original", exist_ok=True)
        os.makedirs("interpolated", exist_ok=True)
        os.makedirs("shifted", exist_ok=True)
        os.makedirs("random", exist_ok=True)

        idx = 0
        for img in render_imgs:
            cv2.imwrite(f"original/rendered_{idx}.png", img)
            idx += 1

        render_imgs = renderer.render_interpolated_views(intrinsics, extrinsics)

        idx = 0
        for img in render_imgs:
            cv2.imwrite(f"interpolated/interpolated_rendered_{idx}.png", img)
            idx += 1
    
        render_imgs = renderer.render_shifted_views(intrinsics, extrinsics)

        idx = 0
        for img in render_imgs:
            cv2.imwrite(f"shifted/shifted_rendered_{idx}.png", img)
            idx += 1

        render_imgs = renderer.render_random_views(intrinsics)            
        idx = 0
        for img in render_imgs:
            cv2.imwrite(f"random/random_rendered_{idx}.png", img)
            idx += 1            
    del renderer
    
if __name__ == '__main__': 
    from option import args
       
    rgbd_processor = RGBD_Processor(args=args, 
                                    cam_idx_list=args.cam_idx_list, 
                                    extrinsic_path=args.extrinsic_path, 
                                    intrinsic_path=args.intrinsic_path,
                                    depth_max=args.max_depth)    

    args.output_path = os.path.join(args.output_dir, args.reconstruct_geometry)
    os.makedirs(args.output_path, exist_ok=True)    
        
    if args.reconstruct_geometry == "color_ptcl":
        reconstruct_ptcl(args, rgbd_processor)
    elif args.reconstruct_geometry == "textured_mesh":
        reconstruct_mesh(args, rgbd_processor)
    else:
        raise ValueError("Invalid mode")