import cv2
import numpy as np
import open3d as o3d
import json
import torch
from moge.model.v1 import MoGeModel
from geometry_processor import Mesh_Processor, Ptcl_Processor
import os
import utils3d
from textured_mapping import *
from tqdm import tqdm
import glob
from mesh_simplify import MeshSimplifier
import time
import numpy as np

import math, numpy as np

# original value in blender aligned to scaniverse 3d
# arena
arena_cam0_loc = (1.7334, 2.3829, -1.2887)            # meters
arena_cam0_eul_deg = (420.6, 18.921, -54.566)         # degrees

# # couch
couch_cam0_loc = (-0.853268, 1.89047, -1.6608)   
couch_cam0_eul_deg = (132.697, -194.743, 50.1609)

# # kitchen
kitchen_cam0_loc = (3.25123, 1.58253, 0.483055)   
kitchen_cam0_eul_deg = (140.259, -375.62, -434.86)

# # whiteboard
whiteboard_cam0_loc = (1.54115, 1.50426, -2.08984)   
whiteboard_cam0_eul_deg = (78.2128, -6.96374, -70.2771)

# def blender_to_world_transform(relative_extrinsics, location, euler_deg_xyz):
#     location = [-x for x in location] #some how need to negate this
#     euler_deg_xyz = [-x for x in euler_deg_xyz] #some how need to negate this
#     rx, ry, rz = np.deg2rad(euler_deg_xyz)

#     def Rx(a): return np.array([
#         [1, 0, 0],
#         [0, math.cos(a), -math.sin(a)],
#         [0, math.sin(a), math.cos(a)]
#     ])
#     def Ry(a): return np.array([
#         [math.cos(a), 0, math.sin(a)],
#         [0, 1, 0],
#         [-math.sin(a), 0, math.cos(a)]
#     ])
#     def Rz(a): return np.array([
#         [math.cos(a), -math.sin(a), 0],
#         [math.sin(a), math.cos(a), 0],
#         [0, 0, 1]
#     ])

#     # 1. Build Blender intrinsic rotation
#     R_blender = Rx(rx) @ Ry(ry)  @ Rz(rz)

#     homog = np.zeros((1, 4))
#     homog[0, 3] = 1
#     homog[0, 0:3] = location
    
#     # 3. Final cam0 c2w matrixhomog
#     homorot = np.eye(4)
#     homorot[:3, :3] = R_blender

#     homofinal = homorot @ homog.T

#     new_trans = homofinal / homofinal[3]

#     final = new_trans[:3]

#     T_cam0_c2w = np.eye(4)
#     T_cam0_c2w[:3, :3] = R_blender
#     T_cam0_c2w[:3, 3] = final.reshape((3,))
    
#     # 4. Apply cam0 pose to all relative extrinsics
#     global_extrinsics = [
#         E_rel @ T_cam0_c2w for E_rel in relative_extrinsics
#     ]

#     return global_extrinsics

def blender_to_world_transform(relative_extrinsics, location, euler_deg_xyz):
    """
    Convert a Blender camera pose (location and Euler angles in degrees) to Open3D world coordinates
    and apply to relative extrinsics.

    Args:
        relative_extrinsics: list of 4×4 np.ndarray extrinsic matrices (relative to camera)
        location: tuple of (x, y, z) in meters (Blender coords)
        euler_deg_xyz: tuple of Euler angles (X, Y, Z) in degrees (Blender rotation)

    Returns:
        List of 4×4 np.ndarray global extrinsic matrices in the world frame.
    """
    # 1. Negate Blender coords (Blender→Open3D)
    loc = -np.array(location, dtype=float)
    #    and invert rotation angles
    angles_rad = -np.radians(euler_deg_xyz)
    rx, ry, rz = angles_rad

    # 2. Define elemental rotation matrices
    def rot_x(theta):
        return np.array([
            [1, 0, 0],
            [0, math.cos(theta), -math.sin(theta)],
            [0, math.sin(theta),  math.cos(theta)]
        ])

    def rot_y(theta):
        return np.array([
            [ math.cos(theta), 0, math.sin(theta)],
            [0, 1, 0],
            [-math.sin(theta), 0, math.cos(theta)]
        ])

    def rot_z(theta):
        return np.array([
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta),  math.cos(theta), 0],
            [0, 0, 1]
        ])

    # 3. Compose rotation: X → Y → Z
    R = rot_x(rx) @ rot_y(ry) @ rot_z(rz)

    # 4. Compute translation: rotate the negated location
    t = R @ loc

    # 5. Build 4×4 homogeneous camera-to-world matrix
    T_cam_world = np.eye(4)
    T_cam_world[:3, :3] = R
    T_cam_world[:3,  3] = t

    # 6. Apply to each relative extrinsic
    return [E_rel @ T_cam_world for E_rel in relative_extrinsics]

class RGBD_Processor:
    def __init__(self, depth_estimation=False, cam_idx_list=[0, 1, 2, 3], 
                intrinsic_path="", 
                extrinsic_path="", 
                align_scaniverse=False):
        self.rectify = True # True if you did not do undistort in capture process
        self.align_scaniverse = align_scaniverse
        self.config = None
        self.calibration_data = None

        self.color_imgs = []
        self.depth_imgs = []

        self.color_imgs_path = []
        self.depth_imgs_path = []

        # self.device_count = 0
        self.cam_idx_list = cam_idx_list
        # self.camera_config = {0:"000954314612", 1:"000329792012", 
        #                         2:"000092320412", 3:"000248792012"}
        
        self.intrinsics = []
        self.optimal_intrinsics = []
        self.distortion = []

        self.mapx = []
        self.mapy = []
        self.intrinsic_path = intrinsic_path
        self.extrinsic_path = extrinsic_path
        
        # self.extrinsics = np.load("../camera_config/global_extrinsics.npy")
        self.extrinsics = np.load(extrinsic_path)
        print(f"Loaded extrinsics from {extrinsic_path}")
        self.extrinsics = self.extrinsics[self.cam_idx_list]

        self.final_cloud = None
    
        self.cpu_device = o3d.core.Device('CPU:0')
        self.gpu_device = o3d.core.Device('CUDA:0')
        
        self.output_filenames = []
        
        self.device = torch.device("cuda")

        self.depth_estimation = depth_estimation
        if self.depth_estimation:
            # Load the model from huggingface hub (or load from local).
            self.model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(self.device)           
        
        # self.load_config("../camera_config/XX_config.json")
        self.load_config(intrinsic_path)
        print(f"Loaded intrinsic from: {intrinsic_path}")        
        self.load_calibration()        
        
        self.geometrys = []
        
    def load_config(self, config_path):
        with open(config_path) as f:
            self.config = json.load(f)
            self.calib_data = self.config["device_calibration"]
            # self.device_count = self.config["kinect_config"]["device_count"]
            self.device_count = len(self.cam_idx_list)
            self.camera_config = self.config["camera_config"]

    def load_calibration(self):
        for cam_idx in self.cam_idx_list:
            cam_serial = self.camera_config[str(cam_idx)]
            intrinsic_param = self.calib_data[cam_serial]["intrinsics"]
            intrinsics = np.identity(3)
            #instrincs are based on 4K (from XX_config.json)
            #for 1080p divide by 2 
            intrinsics[0, 0] = intrinsic_param["fx"] // 2.0 
            intrinsics[1, 1] = intrinsic_param["fy"] // 2.0
            intrinsics[0, 2] = intrinsic_param["cx"] // 2.0
            intrinsics[1, 2] = intrinsic_param["cy"] // 2.0

            distortion_param = self.calib_data[cam_serial]["distortion"]
            distortion = np.array([distortion_param["k1"], distortion_param["k2"], 
                                    distortion_param["p1"], distortion_param["p2"], 
                                    distortion_param["k3"]])

            optimal_intrinsics_param = self.calib_data[cam_serial]["optimal_intrinsics"]
            optimal_intrinsics = np.identity(3)
            optimal_intrinsics[0, 0] = optimal_intrinsics_param["fx"] // 2.0
            optimal_intrinsics[1, 1] = optimal_intrinsics_param["fy"] // 2.0
            optimal_intrinsics[0, 2] = optimal_intrinsics_param["cx"] // 2.0
            optimal_intrinsics[1, 2] = optimal_intrinsics_param["cy"] // 2.0

            self.intrinsics.append(intrinsics)
            self.optimal_intrinsics.append(optimal_intrinsics)
            self.distortion.append(distortion)

            mapx, mapy = cv2.initUndistortRectifyMap(intrinsics, distortion, None, \
                                                        optimal_intrinsics, (1920, 1080), cv2.CV_32FC1)

            self.mapx.append(mapx)
            self.mapy.append(mapy)

            loaded_idx = self.cam_idx_list.index(cam_idx)

            # print(f"Camera {cam_idx} intrinsic: {intrinsics}")
            # print(f"Camera {cam_idx} extrsincis: {self.extrinsics[loaded_idx]}")
        if self.rectify:
            self.intrinsics = self.optimal_intrinsics
     
        if self.align_scaniverse:

            # self.extrinsics = blender_to_world_transform(relative_extrinsics=self.extrinsics, 
            #                                             location=arena_cam0_loc, 
            #                                             euler_deg_xyz=arena_cam0_eul_deg)
            
            # np.save("../camera_config/global_extrinsics_arena_static_scaniverse_aligned.npy", self.extrinsics)
            # np.save("../camera_config/global_extrinsics_arena_scaniverse_aligned.npy", self.extrinsics)
            

                    
            # self.extrinsics = blender_to_world_transform(relative_extrinsics=self.extrinsics, 
            #                                             location=couch_cam0_loc, 
            #                                             euler_deg_xyz=couch_cam0_eul_deg)
            
            # np.save("../camera_config/global_extrinsics_couch_scaniverse_aligned.npy", self.extrinsics)

            # self.extrinsics = blender_to_world_transform(relative_extrinsics=self.extrinsics, 
            #                                             location=kitchen_cam0_loc, 
            #                                             euler_deg_xyz=kitchen_cam0_eul_deg)
            
            # np.save("../camera_config/global_extrinsics_kitchen_scaniverse_aligned.npy", self.extrinsics)                        
            
     
            self.extrinsics = blender_to_world_transform(relative_extrinsics=self.extrinsics, 
                                                        location=whiteboard_cam0_loc, 
                                                        euler_deg_xyz=whiteboard_cam0_eul_deg)
            
            np.save("../camera_config/global_extrinsics_whiteboard_scaniverse_aligned.npy", self.extrinsics)     
        
            for cam_idx in self.cam_idx_list:
                print(f"Camera {cam_idx} scaniverse extrsincis: {self.extrinsics[cam_idx]}")
        
            exit(1)
        # for i, E in enumerate(self.extrinsics):
        #     print(f"Camera {self.cam_idx_list[i]} extrinsics (world):\n{E}\n")
        
    def get_calibration(self):
        return self.intrinsics, self.optimal_intrinsics, self.distortion, self.extrinsics
    
    def get_images(self):
        return self.color_imgs, self.depth_imgs
    
    def get_rgbds_for_cam(self, cam_idx):
        rgb_imgs_for_cam = [frame[cam_idx] for frame in self.color_imgs]
        depth_imgs_for_cam = [frame[cam_idx] for frame in self.depth_imgs]
        return rgb_imgs_for_cam, depth_imgs_for_cam
    
    # load images for each timestamp [rgb*device_count, depth*device_count]
    def load_image(self, data_path, frame_idx=0, only_depth=False, cbf=False, load_memory=True):
        color_imgs = []
        depth_imgs = []
        
        for cam_idx in self.cam_idx_list:
            rgb_path = f'{data_path}/cam{cam_idx}/rgb/{frame_idx}.png'
            depth_path = f'{data_path}/cam{cam_idx}/trans_depth/{frame_idx}_depth.png'
            
            if not load_memory:
                color_imgs.append(rgb_path)
                depth_imgs.append(depth_path)
                continue
            
            self.color_imgs_path.append(rgb_path)
            self.depth_imgs_path.append(depth_path)
            
            # rgb_path = f'{data_path}/cam{cam_idx}/rgb/*_{frame_idx}.png'
            # rgb_files = glob.glob(rgb_path)
            # assert(len(rgb_files) == 1), f"Error: {len(rgb_files)} files found for {rgb_path}"
            # rgb_path = rgb_files[0]        
            # depth_path = f'{data_path}/cam{cam_idx}/trans_depth/*_{frame_idx}_depth.png'
            # depth_files = glob.glob(depth_path)
            # assert(len(depth_files) == 1), f"Error: {len(depth_files)} files found for {depth_path}"
            # depth_path = depth_files[0]
            if cbf:
                depth_path = f'{data_path}/cam{cam_idx}/cbf_depth/{frame_idx}_depth.png'
            
            #print(f"Loading frame {rgb_path}, {depth_path}")
            if not only_depth:
                color_img = cv2.imread(f"{rgb_path}")
                color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            else:
                color_img = None            
            depth_img = cv2.imread(f"{depth_path}", cv2.IMREAD_ANYDEPTH)
                        
            if self.depth_estimation:
                output = self.run_with_gpu(color_img)
                points, output_depth, mask = output['points'], output['depth'], output['mask']
                edge_mask = utils3d.numpy.depth_edge(output_depth, rtol=0.03, mask=mask)
                output_depth = np.where(edge_mask, 0, output_depth)    
                
                output_depth = output_depth / np.max(output_depth)
                depth_img = output_depth * np.max(depth_img)
                        
            # if self.rectify:
            #     color_img = cv2.remap(color_img, self.mapx[cam_idx], self.mapy[cam_idx], cv2.INTER_LINEAR)
            #     depth_img = cv2.remap(depth_img, self.mapx[cam_idx], self.mapy[cam_idx], cv2.INTER_NEAREST)

            #     color_imgs.append(color_img)
            #     depth_imgs.append(depth_img)
            # else:
            color_imgs.append(color_img)
            depth_imgs.append(depth_img)

        return color_imgs, depth_imgs #[rgb*device_count], [depth*device_count]
        
    def load_images(self, data_path, test_idx=None, frame_count=0, only_depth=False, cbf=False, load_memory=True):
        for frame_idx in tqdm(range(1, 1 + frame_count), desc="Processing frames"):
            if test_idx and frame_idx != test_idx:
                continue
            color_imgs, depth_imgs = \
                self.load_image(data_path=data_path, frame_idx=frame_idx, only_depth=only_depth, cbf=cbf, load_memory=load_memory)
            self.color_imgs.append(color_imgs)
            self.depth_imgs.append(depth_imgs)
        return self.color_imgs, self.depth_imgs #[rgb*device_count]*frame_count, [depth*device_count]*frame_count
        
    def depth_to_ptcl(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        reconstructor = Ptcl_Processor()
        for frame_idx, (color_imgs, depth_imgs) in tqdm(enumerate(zip(self.color_imgs, self.depth_imgs)),
                                                        total=len(self.depth_imgs), 
                                                        desc="depth to ptcl"):
            fused_cloud = o3d.geometry.PointCloud()            
            for cam_idx in self.cam_idx_list:
                loaded_idx = self.cam_idx_list.index(cam_idx)
                # if self.rectify:
                #     pc = reconstructor.depth_to_ptcl(
                #                                 depth_img=depth_imgs[loaded_idx],
                #                                 intrinsic=self.optimal_intrinsics[loaded_idx],
                #                                 extrinsics=self.extrinsics[loaded_idx])
                # else:
                pc = reconstructor.depth_to_ptcl(
                                            depth_img=depth_imgs[loaded_idx],
                                            intrinsic=self.intrinsics[loaded_idx],
                                            extrinsics=self.extrinsics[loaded_idx])
                fused_cloud += pc.to_legacy()
                
            # Compute multi-view UV mapping.
            # fused_cloud = compute_color_mapping(fused_cloud, self.intrinsics, self.extrinsics, depth_imgs, color_imgs)   
            # self.geometrys.append(fused_cloud)
            reconstructor.transform_and_write_point_cloud(f"{save_path}/{frame_idx}.ply", fused_cloud)
            self.geometrys.append(f"{save_path}/{frame_idx}.ply")    
            
    def rgbd_to_ptcl(self, save_path, downsample=False):
        os.makedirs(save_path, exist_ok=True)

        reconstructor = Ptcl_Processor(downsample=downsample)
        for frame_idx, (color_imgs, depth_imgs) in tqdm(enumerate(zip(self.color_imgs, self.depth_imgs)),
                                                        total=len(self.depth_imgs), 
                                                        desc="rgbd to ptcl"):
            fused_cloud = o3d.geometry.PointCloud()            
            for cam_idx in self.cam_idx_list:
                loaded_idx = self.cam_idx_list.index(cam_idx)                
                # if self.rectify:
                #     pc = reconstructor.rgbd_to_ptcl(rgb_img=color_imgs[loaded_idx],
                #                                 depth_img=depth_imgs[loaded_idx],
                #                                 intrinsic=self.optimal_intrinsics[loaded_idx],
                #                                 extrinsics=self.extrinsics[loaded_idx])
                # else:
                pc = reconstructor.rgbd_to_ptcl(rgb_img=color_imgs[loaded_idx],
                                            depth_img=depth_imgs[loaded_idx],
                                            intrinsic=self.intrinsics[loaded_idx],
                                            extrinsics=self.extrinsics[loaded_idx])
                fused_cloud += pc.to_legacy()
                
            reconstructor.transform_and_write_point_cloud(f"{save_path}/{frame_idx}.ply", fused_cloud)
    
    def depth_to_mesh(self, save_path):
        os.makedirs(save_path, exist_ok=True)        
        reconstructor = Mesh_Processor()
        for frame_idx, (color_imgs, depth_imgs) in tqdm(enumerate(zip(self.color_imgs, self.depth_imgs)),
                                                        total=len(self.depth_imgs), 
                                                        desc="depth to mesh"):
            for cam_idx in self.cam_idx_list:
                loaded_idx = self.cam_idx_list.index(cam_idx)                                
                reconstructor.integrate_depth(
                                            depth_imgs[loaded_idx], 
                                            self.intrinsics[loaded_idx],
                                            self.extrinsics[loaded_idx]) 
            mesh_t = reconstructor.extract_mesh()
            mesh = mesh_t.to_legacy()
            mesh.vertex_normals = o3d.utility.Vector3dVector([])
            # t1 = time.time()
            o3d.io.write_triangle_mesh(f"{save_path}/{frame_idx}.obj", mesh, write_vertex_normals=False)
            # print(f"elapsed time for write: {(time.time()-t1)*1000:.2f} ms")
            self.geometrys.append(f"{save_path}/{frame_idx}.obj")

    def depth_to_mesh_fusion(self, save_path):
        meshs = []         
        os.makedirs(save_path, exist_ok=True)
        
        reconstructor = Mesh_Processor()
        for frame_idx, (color_imgs, depth_imgs) in enumerate(zip(self.color_imgs, self.depth_imgs)):
            fused_mesh = o3d.geometry.TriangleMesh()
            for cam_idx in self.cam_idx_list:
                reconstructor.create_voxel_block_grid() #reset            
                loaded_idx = self.cam_idx_list.index(cam_idx)                                
                reconstructor.integrate_depth(
                                            depth_imgs[loaded_idx], 
                                            self.intrinsics[loaded_idx],
                                            self.extrinsics[loaded_idx]) 
                mesh_t = reconstructor.extract_mesh()
                mesh = mesh_t.to_legacy()
                mesh.vertex_normals = o3d.utility.Vector3dVector([])                
                fused_mesh += mesh
            meshs.append(fused_mesh)
            o3d.io.write_triangle_mesh(f"{save_path}/{frame_idx}.obj", fused_mesh, write_vertex_normals=False)
            
        return meshs
    
    def rgbd_to_textured_mesh_use_cpp(self, save_path, decimation_ratio=0.0):
        os.makedirs(save_path, exist_ok=True)
        bin_path = "../build/reconstruct_texture_mesh"
        
        for frame_idx, (color_imgs, depth_imgs) in tqdm(enumerate(zip(self.color_imgs, self.depth_imgs)),
                                                total=len(self.depth_imgs), 
                                                desc="rgbd to textured mesh"):
            out_mesh_path = os.path.join(save_path, f"{frame_idx}.obj")
            
            # print(color_imgs)
            # print(depth_imgs)
                       
            cmd = f"{bin_path} {out_mesh_path} {self.intrinsic_path} {self.extrinsic_path} {int(decimation_ratio*100)} "
            cmd += f"{color_imgs[0]} {depth_imgs[0]} {color_imgs[1]} {depth_imgs[1]} {color_imgs[2]} {depth_imgs[2]} {color_imgs[3]} {depth_imgs[3]}"
            # print(cmd)
            os.system(cmd)

        
    def rgbd_to_textured_mesh(self, save_path, decimation_ratio=0.0, use_python=False):
        os.makedirs(save_path, exist_ok=True)
        reconstructor = Mesh_Processor()        
        
        for frame_idx, (color_imgs, depth_imgs) in tqdm(enumerate(zip(self.color_imgs, self.depth_imgs)),
                                                        total=len(self.depth_imgs), 
                                                        desc="rgbd to textured mesh"):
            # Integrate depth images from all cameras.
            for cam_idx in self.cam_idx_list:
                loaded_idx = self.cam_idx_list.index(cam_idx)                
                
                reconstructor.integrate_depth(
                    depth_imgs[loaded_idx],
                    self.intrinsics[loaded_idx],
                    self.extrinsics[loaded_idx])

            # Extract mesh (assumed to be a t.geometry.TriangleMesh).
            mesh_t = reconstructor.extract_mesh()
            # Convert to legacy mesh so we can assign texture and UV attributes.
            mesh = mesh_t.to_legacy()
            
            if decimation_ratio > 0.0:
                simplifyer = MeshSimplifier(mesh, decimation_ratio=decimation_ratio)
                mesh = simplifyer.simplify()
                
            mesh.vertex_normals = o3d.utility.Vector3dVector([])

            # Create a texture atlas by stitching the color images horizontally.
            stitched_texture = cv2.hconcat(color_imgs)
            
            # Compute multi-view UV mapping.
            mesh = optimized_multi_cam_uv(mesh, self.intrinsics, self.extrinsics, depth_imgs)
            
            # Assign the stitched texture to the mesh.
            texture_o3d = o3d.geometry.Image(stitched_texture)
            mesh.textures = [texture_o3d]
            
            # Save the textured mesh.
            out_mesh_path = os.path.join(save_path, f"{frame_idx}.obj")
            o3d.io.write_triangle_mesh(out_mesh_path, mesh, write_vertex_normals=False)
            # print(f"Textured mesh saved to {out_mesh_path}")
            
    def run_with_gpu(self, rgb_image):
        # https://github.com/microsoft/MoGe/blob/main/moge/scripts/app.py
        image_tensor = torch.tensor(rgb_image, dtype=torch.float32, device=torch.device('cuda')).permute(2, 0, 1) / 255
        output = self.model.infer(image_tensor, apply_mask=True, resolution_level=9)
        output = {k: v.cpu().numpy() for k, v in output.items()}
        return output                        