import open3d as o3d
import numpy as np

meter_to_mm = 1000.0

class Mesh_Processor:
    def __init__(self, in_meters=True, depth_max=10.0, voxel_size=3.0 / 512):
        """ voxel size: default 0.0058 in o3d (0.0058=3/512), 
            This spatial resolution is equivalent to representing 
            a 3m x 3m x 3m (m = meter) room with a dense 512 x 512 x 512 voxel grid. 
        """
        self.voxel_size = voxel_size
        self.block_count = 1000 #default 10000 in o3d
        self.block_resolution = 16 #default in o3d
        
        self.in_meters = in_meters
        if self.in_meters:
            self.depth_scale = meter_to_mm
            self.depth_max = depth_max
        else:        
            self.depth_scale = 1.0 # maintain in mm
            self.depth_max = depth_max * meter_to_mm # in mm
        self.trunc_voxel_multiplier = 8.0 # default in o3d
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
        if not self.in_meters:
            extrinsics *= meter_to_mm # convert to mm        
        depth_img = o3d.t.geometry.Image(depth_img).to(self.compute_device)
        intrinsic_t = o3d.core.Tensor(intrinsic)
        extrinsics_t = o3d.core.Tensor(extrinsics)

        frustum_block_coords = self.voxel_grid.compute_unique_block_coordinates(
                                    depth_img, 
                                    intrinsic_t, extrinsics_t, 
                                    self.depth_scale, self.depth_max)

        self.voxel_grid.integrate(frustum_block_coords, depth_img,
                                    intrinsic_t, extrinsics_t,
                                    self.depth_scale, self.depth_max, 
                                    self.trunc_voxel_multiplier)

    def extract_mesh(self):
        mesh = self.voxel_grid.extract_triangle_mesh(weight_threshold=0.0, estimated_vertex_number=-1).to(self.cpu_device)
        return mesh
    
    
class Ptcl_Processor:
    def __init__(self, in_meters=True, depth_max=10.0, downsample_voxel_size=None):
        self.in_meters = in_meters
        if self.in_meters:
            self.depth_scale = meter_to_mm
            self.depth_max = depth_max # in meter # set this for depth map to remove outliers
        else:        
            self.depth_scale = 1.0 # maintain in mm
            self.depth_max = depth_max * meter_to_mm # in mm        
        self.downsample_voxel_size = downsample_voxel_size
        self.outlier_removal = False
        
    def rgbd_to_ptcl(self, rgb_img: np.ndarray, depth_img: np.ndarray, intrinsic: np.ndarray, extrinsics: np.ndarray):                
        if not self.in_meters:
            extrinsics *= meter_to_mm # convert to mm

        rgbd_img = o3d.t.geometry.RGBDImage(o3d.t.geometry.Image(rgb_img),
                                            o3d.t.geometry.Image(depth_img),
                                            aligned=True)
                    
        pc = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd_img, 
                                                                o3d.core.Tensor(intrinsic), 
                                                                o3d.core.Tensor(extrinsics), 
                                                                depth_scale=self.depth_scale,      
                                                                depth_max=self.depth_max)
        if self.downsample_voxel_size:
            pc = pc.voxel_down_sample(voxel_size=0.005)
        
        return pc
    
    def depth_to_ptcl(self, depth_img: np.ndarray, intrinsic: np.ndarray, extrinsics: np.ndarray):
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
        
        if self.downsample_voxel_size:
            pc = pc.voxel_down_sample(voxel_size=self.downsample_voxel_size)        
        
        if self.outlier_removal:
            pc = pc.remove_statistical_outliers(nb_neighbors=50, std_ratio=0.1)
        
        return pc    