import os, cv2, json
import numpy as np
import open3d as o3d
from tqdm import tqdm
from texture_mapping import optimized_multi_cam_uv
from mesh_simplify import MeshSimplifier
from geometry_processor import Mesh_Processor, Ptcl_Processor

class RGBD_Processor:
    def __init__(self, args, cam_idx_list=[0, 1, 2, 3], 
                intrinsic_path="", 
                extrinsic_path="",
                depth_max = 10.0): # 10.0 meters
        self.args = args
        self.depth_max = depth_max
        self.use_optimal_intrinsic = True

        self.color_imgs = []
        self.depth_imgs = []

        self.cam_idx_list = cam_idx_list

        self.intrinsics = []
        self.optimal_intrinsics = []
        self.distortion = []
        
        self.intrinsic_path = intrinsic_path
        self.extrinsic_path = extrinsic_path
        
        self.extrinsics = np.load(extrinsic_path)
        print(f"Loaded extrinsics from {extrinsic_path}")
        self.extrinsics = self.extrinsics[self.cam_idx_list]
    
        self.cpu_device = o3d.core.Device('CPU:0')
        self.gpu_device = o3d.core.Device('CUDA:0')
                       
        self.load_config(intrinsic_path)
        print(f"Loaded intrinsic from: {intrinsic_path}")        
        self.load_calibration()        
        
        self.geometry_paths = []
        
        self.ptcl_processor = Ptcl_Processor(depth_max=self.depth_max, downsample_voxel_size=args.ptcl_downsample_voxel_size)
        self.mesh_processor = Mesh_Processor(depth_max=self.depth_max, voxel_size=args.mesh_voxel_size,)

        
    def load_config(self, config_path):
        with open(config_path) as f:
            self.config = json.load(f)
            self.calib_data = self.config["device_calibration"]
            self.device_count = len(self.cam_idx_list)
            self.camera_config = self.config["camera_config"]

    def load_calibration(self):
        for cam_idx in self.cam_idx_list:
            cam_serial = self.camera_config[str(cam_idx)]
            intrinsic_param = self.calib_data[cam_serial]["intrinsics"]
            intrinsics = np.identity(3)
            #instrincs are based on 4K 
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

        if self.use_optimal_intrinsic:
            self.intrinsics = self.optimal_intrinsics
        
    def get_calibration(self):
        return self.intrinsics, self.optimal_intrinsics, self.distortion, self.extrinsics
    
    def get_images(self):
        return self.color_imgs, self.depth_imgs
    
    def get_rgbds_for_cam(self, cam_idx):
        rgb_imgs_for_cam = [frame[cam_idx] for frame in self.color_imgs]
        depth_imgs_for_cam = [frame[cam_idx] for frame in self.depth_imgs]
        return rgb_imgs_for_cam, depth_imgs_for_cam
    
    # load images for each timestamp [rgb*device_count, depth*device_count]
    def load_image(self, data_path, frame_idx=0, load_memory=True):
        color_imgs = []
        depth_imgs = []
        
        for cam_idx in self.cam_idx_list:
            rgb_path = f'{data_path}/cam{cam_idx}/rgb/{frame_idx}.png'
            depth_path = f'{data_path}/cam{cam_idx}/trans_depth/{frame_idx}_depth.png'
            
            if not load_memory:
                color_imgs.append(rgb_path)
                depth_imgs.append(depth_path)
                continue
                    
            color_img = cv2.imread(f"{rgb_path}")
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            depth_img = cv2.imread(f"{depth_path}", cv2.IMREAD_ANYDEPTH)
                        
            color_imgs.append(color_img)
            depth_imgs.append(depth_img)

        return color_imgs, depth_imgs 
        
    def load_images(self, data_path, test_idx=None, frame_count=0, load_memory=True):
        for frame_idx in tqdm(range(1, 1 + frame_count), desc="Processing frames"):
            if test_idx and frame_idx != test_idx:
                continue
            color_imgs, depth_imgs = \
                self.load_image(data_path=data_path, frame_idx=frame_idx, load_memory=load_memory)
            self.color_imgs.append(color_imgs)
            self.depth_imgs.append(depth_imgs)
        return self.color_imgs, self.depth_imgs #[rgb*device_count]*frame_count, [depth*device_count]*frame_count
        
    def depth_to_ptcl(self, save_path, ptcl_downsample_voxel_size):
        os.makedirs(save_path, exist_ok=True)

        for frame_idx, (_, depth_imgs) in tqdm(enumerate(zip(self.color_imgs, self.depth_imgs)),
                                                        total=len(self.depth_imgs), 
                                                        desc="depth to ptcl"):
            fused_cloud = o3d.geometry.PointCloud()            
            for cam_idx in self.cam_idx_list:
                loaded_idx = self.cam_idx_list.index(cam_idx)
                pc = self.ptcl_processor.depth_to_ptcl(
                                            depth_img=depth_imgs[loaded_idx],
                                            intrinsic=self.intrinsics[loaded_idx],
                                            extrinsics=self.extrinsics[loaded_idx])
                fused_cloud += pc.to_legacy()
            file_path = f"{save_path}/{frame_idx}.ply"  
            o3d.io.write_point_cloud(file_path, fused_cloud, write_ascii=True)
            self.geometry_paths.append(file_path)    
            
    def rgbd_to_ptcl(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        for frame_idx, (color_imgs, depth_imgs) in tqdm(enumerate(zip(self.color_imgs, self.depth_imgs)),
                                                        total=len(self.depth_imgs), 
                                                        desc="rgbd to ptcl"):
            fused_cloud = o3d.geometry.PointCloud()            
            for cam_idx in self.cam_idx_list:
                loaded_idx = self.cam_idx_list.index(cam_idx)                
                pc = self.ptcl_processor.rgbd_to_ptcl(rgb_img=color_imgs[loaded_idx],
                                            depth_img=depth_imgs[loaded_idx],
                                            intrinsic=self.intrinsics[loaded_idx],
                                            extrinsics=self.extrinsics[loaded_idx])
                fused_cloud += pc.to_legacy()
            file_path = f"{save_path}/{frame_idx}.ply"
            o3d.io.write_point_cloud(file_path, fused_cloud, write_ascii=True)    
            self.geometry_paths.append(file_path)
    
    def depth_to_mesh(self, save_path):
        os.makedirs(save_path, exist_ok=True)        
        for frame_idx, (_, depth_imgs) in tqdm(enumerate(zip(self.color_imgs, self.depth_imgs)),
                                                        total=len(self.depth_imgs), 
                                                        desc="depth to mesh"):
            for cam_idx in self.cam_idx_list:
                loaded_idx = self.cam_idx_list.index(cam_idx)                                
                self.mesh_processor.integrate_depth(
                                            depth_imgs[loaded_idx], 
                                            self.intrinsics[loaded_idx],
                                            self.extrinsics[loaded_idx]) 
            mesh_t = self.mesh_processor.extract_mesh()
            mesh = mesh_t.to_legacy()
            mesh.vertex_normals = o3d.utility.Vector3dVector([])
            file_path = f"{save_path}/{frame_idx}.obj"
            o3d.io.write_triangle_mesh(file_path, mesh, write_vertex_normals=False)
            self.geometry_paths.append(file_path)

    def depth_to_mesh_fusion(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        
        reconstructor = Mesh_Processor(depth_max=self.depth_max)
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
            file_path = f"{save_path}/{frame_idx}.obj"
            o3d.io.write_triangle_mesh(file_path, fused_mesh, write_vertex_normals=False)
            self.geometry_paths.append(file_path)
            
    def rgbd_to_textured_mesh_use_cpp(self, save_path, decimation_ratio=0.0):
        os.makedirs(save_path, exist_ok=True)
        bin_path = "../build/reconstruct_texture_mesh"
        
        for frame_idx, (color_imgs, depth_imgs) in tqdm(enumerate(zip(self.color_imgs, self.depth_imgs)),
                                                total=len(self.depth_imgs), 
                                                desc="rgbd to textured mesh"):
            file_path = f"{save_path}/{frame_idx}.obj"
            # print(color_imgs)
            # print(depth_imgs)
                       
            cmd = f"{bin_path} {file_path} {self.intrinsic_path} {self.extrinsic_path} {int(decimation_ratio*100)} "
            cmd += f"{color_imgs[0]} {depth_imgs[0]} {color_imgs[1]} {depth_imgs[1]} {color_imgs[2]} {depth_imgs[2]} {color_imgs[3]} {depth_imgs[3]}"
            # print(cmd)
            os.system(cmd)
            self.geometry_paths.append(file_path)

    def rgbd_to_textured_mesh(self, save_path, decimation_ratio=0.0):
        os.makedirs(save_path, exist_ok=True)
        for frame_idx, (color_imgs, depth_imgs) in tqdm(enumerate(zip(self.color_imgs, self.depth_imgs)),
                                                        total=len(self.depth_imgs), 
                                                        desc="rgbd to textured mesh"):
            # Integrate depth images from all cameras.
            for cam_idx in self.cam_idx_list:
                loaded_idx = self.cam_idx_list.index(cam_idx)                
                
                self.mesh_processor.integrate_depth(
                    depth_imgs[loaded_idx],
                    self.intrinsics[loaded_idx],
                    self.extrinsics[loaded_idx])

            # Extract mesh (assumed to be a t.geometry.TriangleMesh).
            mesh_t = self.mesh_processor.extract_mesh()
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
            file_path = f"{save_path}/{frame_idx}.obj"
            o3d.io.write_triangle_mesh(file_path, mesh, write_vertex_normals=False)
            self.geometry_paths.append(file_path)   