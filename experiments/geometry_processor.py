import open3d as o3d
import numpy as np
from typing import List, Any
import time
class Mesh_Processor:
    def __init__(self, in_meters=True):
        """ voxel size: default 0.0058 in o3d (0.0058=3/512), 
            This spatial resolution is equivalent to representing 
            a 3m x 3m x 3m (m = meter) room with a dense 512 x 512 x 512 voxel grid. 
        """
        self.voxel_size = 0.01 #3.0 / 512 #0.01
        self.block_count = 1000 #default 10000 in o3d
        self.block_resolution = 16 #default in o3d
        
        self.in_meters = in_meters
        if self.in_meters:
            self.depth_scale = 1000.0 
            self.depth_max = 10.0
        else:        
            self.depth_scale = 1.0 # maintain in mm
            self.depth_max = 10000.0 # in mm
        self.trunc_voxel_multiplier = 8.0 #default in o3d
        self.cpu_device = o3d.core.Device('CPU:0')
        self.gpu_device = o3d.core.Device('CUDA:0')
        self.compute_device = self.cpu_device

        self.voxel_grid = None
        self.create_voxel_block_grid()

    def create_voxel_block_grid(self):
        self.voxel_grid = o3d.t.geometry.VoxelBlockGrid(
                        attr_names=('tsdf', 'weight'),
                        attr_dtypes=(o3d.core.float32, o3d.core.float32),
                        attr_channels=((1), (1)),
                        voxel_size = self.voxel_size,
                        block_resolution = self.block_resolution,
                        block_count = self.block_count,
                        device = self.compute_device)

    def integrate_depth(self, depth_img: np.ndarray, intrinsic: np.ndarray, extrinsics: np.ndarray):
        # rgb_img = o3d.t.geometry.Image(rgb_img)
        
        if not self.in_meters:
            extrinsics *= 1000.0 # convert to mm        
        # t1 = time.time()
        depth_img = o3d.t.geometry.Image(depth_img).to(self.compute_device)
        # print(f"elapsed time for depth_img: {(time.time()-t1)*1000:.2f} ms")
        intrinsic_t = o3d.core.Tensor(intrinsic)
        extrinsics_t = o3d.core.Tensor(extrinsics)
        # t1 = time.time()
        frustum_block_coords = self.voxel_grid.compute_unique_block_coordinates(
                                    depth_img, 
                                    intrinsic_t, extrinsics_t, 
                                    self.depth_scale, self.depth_max)

        self.voxel_grid.integrate(frustum_block_coords, depth_img,
                                    intrinsic_t, extrinsics_t,
                                    self.depth_scale, self.depth_max, 
                                    self.trunc_voxel_multiplier)
        # print(f"elapsed time for integrate: {(time.time()-t1)*1000:.2f} ms")

    def extract_mesh(self):
        # t1 = time.time()
        mesh = self.voxel_grid.extract_triangle_mesh(weight_threshold=0.0, estimated_vertex_number=-1).to(self.cpu_device)
        # print(f"elapsed time for extract: {(time.time()-t1)*1000:.2f} ms")
        #ply = self.voxel_grid.extract_point_cloud()
        return mesh
    
    
class Ptcl_Processor:
    def __init__(self, in_meters=True, downsample=False):
        self.in_meters = in_meters
        if self.in_meters:
            self.depth_scale = 1000.0 
            self.depth_max = 7.0 #in meter # set this for depth map to remove outliers
        else:        
            self.depth_scale = 1.0 # maintain in mm
            self.depth_max = 70000.0 # in mm        
        self.downsample = downsample
        self.background_segmentation = False
        self.crop_center = False
        self.outlier_removal = False

    def transform_and_write_point_cloud(self, filename, pcd):
        # pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        o3d.io.write_point_cloud(filename, pcd, write_ascii=True, compressed=True, print_progress=True)
        
    def rgbd_to_ptcl(self, rgb_img: np.ndarray, depth_img: np.ndarray, intrinsic: np.ndarray, extrinsics: np.ndarray):                
        if self.background_segmentation:
            # limit 3.5m to 100m
            depth_img = np.where((depth_img >= 3500.0) & (depth_img <= 1000000.0), depth_img, 0)
        elif self.crop_center:
            self.depth_max = 3.0

        if not self.in_meters:
            extrinsics *= 1000.0 # convert to mm

        rgbd_img = o3d.t.geometry.RGBDImage(o3d.t.geometry.Image(rgb_img),
                                            o3d.t.geometry.Image(depth_img),
                                            aligned=True)
                    
        pc = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd_img, 
                                                                o3d.core.Tensor(intrinsic), 
                                                                o3d.core.Tensor(extrinsics), 
                                                                depth_scale=self.depth_scale,      
                                                                depth_max=self.depth_max)

        if self.downsample:
            pc = pc.voxel_down_sample(voxel_size=0.005)
        
        return pc
    
    def depth_to_ptcl(self, depth_img: np.ndarray, intrinsic: np.ndarray, extrinsics: np.ndarray):
        if self.background_segmentation:
            # limit 3.5m to 100m
            depth_img = np.where((depth_img >= 3500.0) & (depth_img <= 1000000.0), depth_img, 0)
        elif self.crop_center:
            self.depth_max = 3.0

        if not self.in_meters:
            extrinsics *= 1000.0 # convert to mm
         
        depth_o3d = o3d.t.geometry.Image(depth_img)
        pc = o3d.t.geometry.PointCloud.create_from_depth_image(
            depth_o3d,
            o3d.core.Tensor(intrinsic),
            o3d.core.Tensor(extrinsics),
            depth_scale=self.depth_scale,
            depth_max=self.depth_max, 
            with_normals=False) 
        
        if self.downsample:
            pc = pc.voxel_down_sample(voxel_size=0.01)        
        
        if self.outlier_removal:
            pc = pc.remove_statistical_outliers(nb_neighbors=50, std_ratio=0.1)
        
        return pc    