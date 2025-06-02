from rgbd_processor import RGBD_Processor
import os
import numpy as np
from ffmpeg import *
import open3d as o3d
from option import args
from hue import encode_hue, decode_hue
from simple_packing import depth_to_y, y_to_depth
from triangle import triangle_encode, triangle_decode
from simple_packing import *
from yuv_io import write_frame_to_yuv, read_frame_from_yuv, write_frame_to_y, read_frame_from_y
from metrics import *
import cv2
from rgbd_enc_dec import *
from draco import Draco
from geometry_metrics import *
import glob
from multiview_renderer import *

def reconstruct(args, geometry, cam_idx_list=[0,1,2,3], only_depth=False):    
    rgbd_processor = RGBD_Processor(cam_idx_list=cam_idx_list, 
                                    extrinsic_path=args.extrinsic_path, 
                                    intrinsic_path=args.intrinsic_path)
    if geometry == "textured_mesh":
        rgbd_processor.load_images(data_path=args.data_path, test_idx=args.test_idx, frame_count=args.frame_count, only_depth=only_depth, cbf=args.cbf, load_memory=False)        
    else:
        rgbd_processor.load_images(data_path=args.data_path, test_idx=args.test_idx, frame_count=args.frame_count, only_depth=only_depth, cbf=args.cbf)
    
    if geometry == "color_ptcl":
        rgbd_processor.rgbd_to_ptcl(save_path=f"{args.output_path}", downsample=args.downsample)
    elif geometry == "ptcl":
        rgbd_processor.depth_to_ptcl(save_path=f"{args.output_path}")
    elif geometry == "textured_mesh":
        if args.decimation_ratio > 0.0:
            save_path = f"{args.output_path}_decimated_{args.decimation_ratio}"
        else:
            save_path = args.output_path
        # rgbd_processor.rgbd_to_textured_mesh(save_path=save_path, decimation_ratio=args.decimation_ratio)        
        rgbd_processor.rgbd_to_textured_mesh_use_cpp(save_path=save_path, decimation_ratio=args.decimation_ratio)
    elif geometry == "mesh":
        if args.decimation_ratio > 0.0:
            save_path = f"{args.output_path}_decimated_{args.decimation_ratio}"
        else:
            save_path = args.output_path        
        rgbd_processor.depth_to_mesh(save_path=save_path, decimation_ratio=args.decimation_ratio)  
    
    else:
        raise ValueError("Invalid representation")
    
    return rgbd_processor.geometrys
    
def pipeline_ptcl(args):
    args.output_path = os.path.join(args.output_dir, args.mode, args.reconstruct_geometry, f"qp{args.draco_qp}")        
    os.makedirs(args.output_path, exist_ok=True)    
    sizes = []
    distortions = []
    log_txt = "size(Byte) \t distortion(AHD)\n"
    
    draco = Draco(exec_dir=args.exec_dir, qp=args.draco_qp, convert=True, color=args.color, ptcl=True)    
    for geometry_filepath in args.gt_geometrys:
        # Extract the filename without the extension
        filename = os.path.splitext(os.path.basename(geometry_filepath))[0]
        print(f"Processing {geometry_filepath}, {args.output_path}/{filename}.drc")
        
        encoded_filepath = f"{args.output_path}/{filename}.drc"
        draco.encode(input_path=f"{geometry_filepath}", output_path=encoded_filepath)
        
        decoded_filepath = f"{args.output_path}/{filename}_decoded.ply"        
        draco.decode(input_path=f"{args.output_path}/{filename}.drc", output_path=decoded_filepath)
        
        # Read the ground truth mesh
        gt_ptcl = o3d.io.read_point_cloud(geometry_filepath)
        
        # Convert points to a NumPy array
        # points = np.asarray(gt_ptcl.points)

        # # Calculate the range for each axis
        # x_min, x_max = points[:, 0].min(), points[:, 0].max()
        # y_min, y_max = points[:, 1].min(), points[:, 1].max()
        # z_min, z_max = points[:, 2].min(), points[:, 2].max()

        # # Print the ranges
        # print(f"X range: {x_min} to {x_max}")
        # print(f"Y range: {y_min} to {y_max}")
        # print(f"Z range: {z_min} to {z_max}")        
        
        decoded_ptcl = o3d.io.read_point_cloud(decoded_filepath)
        dist = avg_hausdorff_distance(src=gt_ptcl, tgt=decoded_ptcl) * 1000.0 #to mm        
        file_size = os.path.getsize(encoded_filepath)
        
        sizes.append(file_size)
        distortions.append(dist)
        log_txt += f"{file_size}\t {dist}\n"
        # print(dist)
    f = open(f"{args.output_path}/rd_log.txt", "w")
    
    f.write(f"avg size: {np.mean(np.asarray(sizes)):.3f}, distortion: {np.mean(np.asarray(distortions)):.3f}\n")
    f.write(log_txt)
    

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
    
    #test_multiview_ssim(args)
        
    os.makedirs(args.output_dir, exist_ok=True)    
    args.output_path = os.path.join(args.output_dir, "gt", args.reconstruct_geometry)
    os.makedirs(args.output_path, exist_ok=True)
    
    # gt_geometrys = glob.glob(f"{args.output_path}/*")
    # # if len(gt_geometrys) < args.frame_count:
    gt_geometrys = reconstruct(args, geometry=args.reconstruct_geometry, cam_idx_list=args.cam_idx_list)
    
    if args.mode == "ptcl":
        pipeline_ptcl(args)
    elif args.mode == "mesh":
        pipeline_mesh(args)
    else:
        raise ValueError("Invalid mode")