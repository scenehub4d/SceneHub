import open3d as o3d
import numpy as np
import cv2
from scipy.spatial.transform import Rotation, Slerp
from rgbd_processor import RGBD_Processor
import os

def shift_extrinsic(extrinsic: np.ndarray,
                    direction: str,
                    amount: float = 0.1) -> np.ndarray:
    """
    Return a new 4×4 extrinsic that is shifted in camera‐local coordinates.
    
    Args:
        extrinsic (np.ndarray): Original world→camera 4×4 matrix.
        direction (str): One of "up", "down", "left", or "right".
        amount (float): Distance to move in meters.
    
    Returns:
        np.ndarray: The shifted world→camera extrinsic.
    
    Camera‐local axes:
      • +X points right  
      • +Y points down  
      • +Z points forward
    """
    # Build the camera‐frame translation vector
    if direction == "up":
        v_cam = np.array([0, -amount, 0])
    elif direction == "down":
        v_cam = np.array([0,  amount, 0])
    elif direction == "left":
        v_cam = np.array([-amount, 0, 0])
    elif direction == "right":
        v_cam = np.array([ amount, 0, 0])
    else:
        raise ValueError(f"Unknown direction '{direction}'")

    # Ensure a NumPy array
    E = np.array(extrinsic)
    # Compute camera→world
    E_wc = np.linalg.inv(E)
    # Build a 4×4 T_cam in camera coords
    T_cam = np.eye(4)
    T_cam[:3, 3] = v_cam
    # Apply shift in camera→world, then invert back
    E_wc_shifted = E_wc @ T_cam
    E_shifted    = np.linalg.inv(E_wc_shifted)
    return E_shifted

def interpolate_extrinsics(e1, e2, alpha: float = 0.5) -> np.ndarray:
    """
    Interpolate between two 4x4 extrinsic matrices.
    
    Uses Slerp for the rotation part and linear interpolation for the translation.
    
    Args:
        e1 (list or np.ndarray): First extrinsic matrix (4x4).
        e2 (list or np.ndarray): Second extrinsic matrix (4x4).
        alpha (float): Interpolation factor between 0 and 1.
    
    Returns:
        np.ndarray: The interpolated extrinsic matrix (4x4).
    """
    # Convert inputs to numpy arrays if they are lists.
    e1 = np.array(e1)
    e2 = np.array(e2)
    
    # Extract rotation matrices and translation vectors.
    R1, R2 = e1[:3, :3], e2[:3, :3]
    t1, t2 = e1[:3, 3], e2[:3, 3]
    
    # Use Slerp for rotation interpolation.
    key_rots = Rotation.from_matrix([R1, R2])
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)
    R_interp = slerp([alpha]).as_matrix()[0]
    
    # Linear interpolation for translation.
    t_interp = (1 - alpha) * t1 + alpha * t2
    
    # Assemble the interpolated 4x4 extrinsic matrix.
    E_new = np.eye(4)
    E_new[:3, :3] = R_interp
    E_new[:3, 3] = t_interp
    return E_new

def interpolate_intrinsics(i1, i2, alpha: float = 0.5) -> np.ndarray:
    """
    Interpolate between two 3x3 intrinsic matrices by linearly blending the parameters.
    
    Args:
        i1 (list or np.ndarray): First intrinsic matrix (3x3).
        i2 (list or np.ndarray): Second intrinsic matrix (3x3).
        alpha (float): Interpolation factor between 0 and 1 (default is 0.5).
        
    Returns:
        np.ndarray: The interpolated intrinsic matrix (3x3).
    """
    # Convert inputs to NumPy arrays with a float dtype (ensuring proper numeric type)
    i1 = np.array(i1, dtype=float)
    i2 = np.array(i2, dtype=float)

    fx = (1 - alpha) * i1[0, 0] + alpha * i2[0, 0]
    fy = (1 - alpha) * i1[1, 1] + alpha * i2[1, 1]
    cx = (1 - alpha) * i1[0, 2] + alpha * i2[0, 2]
    cy = (1 - alpha) * i1[1, 2] + alpha * i2[1, 2]
    
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]])


def sample_camera_positions_on_sphere(mesh: o3d.geometry.TriangleMesh, num_samples: int) -> list:
        """
        Randomly sample camera positions on a sphere enclosing the mesh's axis-aligned bounding box.
        Returns a list of 3D positions.
        """
        bbox = mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = bbox.get_extent()
        # radius is half the diagonal of the bounding box
        radius = np.linalg.norm(extent) / 2.0
        positions = []
        for _ in range(num_samples):
            z = np.random.uniform(-1.0, 1.0)
            theta = np.random.uniform(0.0, 2.0 * np.pi)
            r = np.sqrt(1.0 - z * z)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            positions.append(center + radius * np.array([x, y, z]))
        return positions
    

def look_at_extrinsic(camera_pos, target, world_up=np.array([0,1,0])):
    # 1) forward in world (camera’s +Z)
    f = (target - camera_pos)
    f /= np.linalg.norm(f)

    # 2) right = world_up × forward  (→ camera’s +X)
    r = np.cross(world_up, f)
    r /= np.linalg.norm(r)

    # 3) up    = forward × right    (→ camera’s +Y)
    u = np.cross(f, r)

    E_c2w = np.eye(4)
    E_c2w[:3, :3] = np.stack([u, r, f], axis=1)  # rotation cols = X,Y,Z axes
    E_c2w[:3,  3] = camera_pos                   # translation = camera center

    # 5) invert to get world→camera extrinsic
    E_w2c = np.linalg.inv(E_c2w)

    return E_w2c

# def sample_camera_extrinsics_on_sphere_pcd(
#     pcd: o3d.geometry.PointCloud,
#     num_samples: int,
#     use_tight_sphere: bool = True
# ) -> list:
#     bbox   = pcd.get_axis_aligned_bounding_box()
#     center = np.asarray(pcd.points).mean(axis=0) if use_tight_sphere else bbox.get_center()

#     if use_tight_sphere:
#         pts    = np.asarray(pcd.points)
#         radius = np.linalg.norm(pts - center, axis=1).max()
#     else:
#         extent = bbox.get_extent()
#         # original: radius = np.linalg.norm(extent)/2.0
#         # alternative “face-sphere”:
#         radius = extent.max() / 2.0

#     extrinsics = []
#     for _ in range(num_samples):
#         z, θ = np.random.uniform(-1,1), np.random.uniform(0,2*np.pi)
#         r_xy = np.sqrt(1 - z*z)
#         pos  = center + radius * np.array([r_xy*np.cos(θ), r_xy*np.sin(θ), z])
#         E    = look_at_extrinsic(pos, target=center)
#         extrinsics.append(E)
#     return extrinsics

# def sample_camera_extrinsics_inside_bbox(
#     pcd: o3d.geometry.PointCloud,
#     num_samples: int
# ) -> list:
#     """
#     Uniformly sample num_samples camera extrinsics with
#     positions inside the point-cloud’s AABB, each looking
#     at the AABB’s center.
#     """
#     aabb = pcd.get_axis_aligned_bounding_box()
#     mn, mx = aabb.min_bound, aabb.max_bound
#     center = aabb.get_center()

#     extrinsics = []
#     for _ in range(num_samples):
#         eye = np.array([
#             np.random.uniform(mn[0], mx[0]),
#             np.random.uniform(mn[1], mx[1]),
#             np.random.uniform(mn[2], mx[2]),
#         ])
#         extrinsics.append( look_at_extrinsic(eye, center) )
#     return extrinsics

def sample_camera_extrinsics_inside_bbox_mesh(
    mesh: o3d.geometry.PointCloud,
    num_samples: int,
    original_ext: list,
    random_seed: int = 42
) -> list:
    """
    Uniformly sample num_samples camera extrinsics with
    positions inside the point-cloud’s AABB, each looking
    at the AABB’s center.
    """
    np.random.seed(random_seed)
    
    aabb = mesh.get_axis_aligned_bounding_box()
    raw_min    = aabb.get_min_bound()  # [xmin0, ymin0, zmin0]
    raw_max    = aabb.get_max_bound()  # [xmax0, ymax0, zmax0]

    # extent = raw_max - raw_min  
    # #print(f"AABB min: {raw_min}, max: {raw_max}")
    # print(f"Extent (X, Y, Z): {extent}")
    # 1) Your hard “world” limits for x, y, z:
    lower = np.array([-5, -0.5, -5])
    upper = np.array([5, 5, 5])

    # 3) Clamp both min and max into [lower, upper]:
    clamped_min = np.clip(raw_min, lower, upper)
    clamped_max = np.clip(raw_max, lower, upper)
    center = aabb.get_center()

    extrinsics = []
    for _ in range(num_samples):
        # eye = np.array([
        #     np.random.uniform(mn[0], mx[0]),
        #     np.random.uniform(mn[1], mx[1]),
        #     np.random.uniform(mn[2], mx[2]),
        # ])
        eye = np.random.uniform(clamped_min, clamped_max)
        # x = original_ext[0, 3]
        # y = original_ext[1, 3]
        # z = original_ext[2, 3]
        # eye  = original_ext[0, :3, 3]
        # print(eye.shape)
        extrinsics.append( look_at_extrinsic(eye, center) )
    # np.save(f"random_extrinsics_seed{random_seed}_sample{num_samples}.npy", np.array(extrinsics))
    # exit(1)        
    return extrinsics


def filter_by_angle_threshold(extrinsics: list, ref_extrinsic: np.ndarray, max_deg: float) -> list:
    """
    Filter a list of extrinsic matrices, keeping only those whose camera
    forward direction deviates from the reference by <= max_deg degrees.
    """
    def view_direction(E: np.ndarray) -> np.ndarray:
        # camera z-axis in world coords is the third column of R^T
        return E[:3, :3].T[:, 2]

    ref_dir = view_direction(ref_extrinsic)
    selected = []
    for E in extrinsics:
        v = view_direction(E)
        cosang = np.clip(np.dot(ref_dir, v) / (np.linalg.norm(ref_dir) * np.linalg.norm(v)), -1.0, 1.0)
        ang = np.degrees(np.arccos(cosang))
        if ang <= max_deg:
            selected.append(E)
    return selected


# Utility to add camera spheres
def make_circle_cloud(center, radius, n_pts=200, normal=np.array([0, 0, 1])):
    center = np.asarray(center, dtype=float)
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    u = np.cross(normal, [0, 0, 1])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(normal, [0, 1, 0])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    theta = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    points = center + radius*np.outer(np.cos(theta), u) + radius*np.outer(np.sin(theta), v)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return pcd

class MultiViewMeshRenderer:
    def __init__(self, mesh_path: str, width: int = 1920, height: int = 1080, random_seed: int = 42, \
            random_extrinsics: list = None):
        """
        Initialize the multiview renderer by loading the scene and mesh once.
        
        Args:
            mesh_path (str): Path to the mesh file.
            width (int): Image width.
            height (int): Image height.
        """
        self.width = width
        self.height = height
        self.random_seed = random_seed
        self.random_extrinsics = random_extrinsics
        # Load the mesh
        self.mesh = o3d.io.read_triangle_mesh(mesh_path, True)
        if not self.mesh.has_vertex_normals():
            self.mesh.compute_vertex_normals()
        
        # Set up the offscreen renderer once
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        
        # Configure the material
        self.material = o3d.visualization.rendering.MaterialRecord()
        self.material.shader = "defaultUnlit"  # Change to "defaultUnlit" if desired
        # self.material.shader = "defaultLit"  # Change to "defaultUnlit" if desired

        self.material.aspect_ratio = 1.0
        
        # Handle texture if the mesh contains one
        if self.mesh.has_textures():
            texture_np = np.asarray(self.mesh.textures[0])
            if texture_np.dtype != np.uint8:
                texture_np = (texture_np * 255).astype(np.uint8)
            # Open3D expects texture origin at the bottom
            texture_np = np.flipud(texture_np)
            texture_np = np.ascontiguousarray(texture_np)
            # Convert grayscale to RGB or RGBA to RGB if necessary
            if texture_np.shape[-1] == 1:
                texture_np = np.repeat(texture_np, 3, axis=-1)
            elif texture_np.shape[-1] == 4:
                texture_np = texture_np[..., :3]
            new_texture = o3d.geometry.Image(texture_np)
            self.material.albedo_img = new_texture
        else:
            print("No texture found")
            assert(0)
        
        
        # Add the mesh to the renderer's scene (this is done only once)
        self.renderer.scene.add_geometry("mesh", self.mesh, self.material)
        self.renderer.scene.scene.set_sun_light([1, 1, 1], [1.0, 1.0, 1.0], 300000)
        self.renderer.scene.scene.enable_sun_light(True)
        # self.renderer.scene.scene.enable_sun_light(False)
        # self.renderer.scene.scene.set_indirect_light_intensity(2.0)

        self.renderer.scene.set_background([0, 0, 0, 1])

    def render_view(self, intrinsic_matrix: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
        """
        Render one view given an intrinsic and extrinsic matrix.
        
        Args:
            intrinsic_matrix (np.ndarray): A 3x3 intrinsic matrix.
            extrinsic (np.ndarray): A 4x4 extrinsic matrix.
        
        Returns:
            np.ndarray: The rendered image in BGR format.
        """
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]
        pinhole_intrinsic = o3d.camera.PinholeCameraIntrinsic(self.width, self.height, fx, fy, cx, cy)
        
        # Update the camera parameters for this view.
        self.renderer.setup_camera(pinhole_intrinsic, extrinsic)
        
        # Render and convert the result to a NumPy array.
        rendered_image = self.renderer.render_to_image()
        rendered_np = np.asarray(rendered_image)
        
        # Convert from RGB (Open3D) to BGR (OpenCV)
        rendered_np_bgr = cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR)
        return rendered_np_bgr

    def render_views(self, intrinsics_list: list, extrinsics_list: list, cam_idx_list: list = None) -> list:
        """
        Render and return a list of images for the given intrinsic and extrinsic matrices.
        
        Args:
            intrinsics_list (list): A list of 3x3 intrinsic matrices (NumPy arrays).
            extrinsics_list (list): A list of 4x4 extrinsic matrices (NumPy arrays).
            cam_idx_list (list): A list of indices for which the view should be rendered.
                If None, all views are rendered.
        
        Returns:
            list: A list of rendered images as NumPy arrays in BGR format.
        """
        if cam_idx_list is None:
            cam_idx_list = list(range(len(intrinsics_list)))
            
        rendered_images = []
        for idx in cam_idx_list:
            intrinsic_matrix = intrinsics_list[idx]
            extrinsic = extrinsics_list[idx]
            
            # Extract intrinsic parameters
            fx = intrinsic_matrix[0, 0]
            fy = intrinsic_matrix[1, 1]
            cx = intrinsic_matrix[0, 2]
            cy = intrinsic_matrix[1, 2]
            pinhole_intrinsic = o3d.camera.PinholeCameraIntrinsic(self.width, self.height, fx, fy, cx, cy)
            
            # Update the camera parameters in the renderer
            self.renderer.setup_camera(pinhole_intrinsic, extrinsic)
            
            # Render and convert the image from Open3D Image to NumPy array (RGB)
            rendered_image = self.renderer.render_to_image()
            rendered_np = np.asarray(rendered_image)
            
            # Convert to BGR for OpenCV saving and display
            rendered_np_bgr = cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR)            
            # Save the rendered image
            # cv2.imwrite(f"rendered_{idx}.png", rendered_np_bgr)
            # print(f"Rendered image saved as rendered_{idx}.png")
            
            rendered_images.append(rendered_np_bgr)
        
        return rendered_images

    def render_interpolated_views(self, intrinsics_list: list, extrinsics_list: list, alpha: float = 0.5) -> list:
        """
        Create novel views by interpolating between each consecutive pair of provided views.
        For n provided views, n novel views are generated by interpolating between view i and view (i+1) mod n.
        
        Args:
            intrinsics_list (list): List of 3x3 intrinsic matrices.
            extrinsics_list (list): List of 4x4 extrinsic matrices.
            alpha (float): Interpolation factor (default 0.5 for midpoints).
            
        Returns:
            list: Rendered novel view images (BGR format).
        """
        n = len(intrinsics_list)
        novel_images = []
        for i in range(n):
            # Interpolate cyclically between view i and view (i+1)
            j = (i + 1) % n
            new_intrinsic = interpolate_intrinsics(intrinsics_list[i], intrinsics_list[j], alpha)
            new_extrinsic = interpolate_extrinsics(extrinsics_list[i], extrinsics_list[j], alpha)
            novel_img = self.render_view(new_intrinsic, new_extrinsic)
            # cv2.imwrite(f"novel_rendered_{i}.png", novel_img)
            # print(f"Saved novel_rendered_{i}.png")
            novel_images.append(novel_img)
        return novel_images
    
    def render_shifted_views(
        self,
        intrinsics_list: list,
        extrinsics_list: list,
        amount: float = 0.1
    ) -> list:
        """
        For each original view, shift its extrinsic up/down/left/right by `amount`
        (in camera‐local coordinates) and render four new images per view.
        """
        directions = ["up", "down", "left", "right"]
        shifted_images = []
        for K, E in zip(intrinsics_list, extrinsics_list):
            for d in directions:
                E_shift = shift_extrinsic(E, d, amount)
                shifted_images.append(self.render_view(K, E_shift))
        return shifted_images 
    
    def render_random_views(
        self,
        intrinsics_list: list,
        num_samples: int = 20
    ) -> list:
        """
        Sample `num_samples` camera extrinsics uniformly inside the mesh’s AABB
        (looking at its center), and render them all. Uses the first intrinsic
        in intrinsics_list for every view.
        """
        # sample new extrinsics using your helper
        
        if self.random_extrinsics is not None:
            random_extrs = self.random_extrinsics
        else:
            random_extrs = sample_camera_extrinsics_inside_bbox_mesh(self.mesh, num_samples, None, self.random_seed)
            # np.save(f"random_extrinsics_seed{self.random_seed}_sample{num_samples}.npy", np.array(random_extrs))
            # exit(1)
        # random_extrs = np.load(f"arena_random_extrinsics_seed{self.random_seed}_sample{num_samples}.npy")
        print(f"random_extrinsics shape: {random_extrs.shape[0]}")
        assert(random_extrs.shape[0] == num_samples)
        # Save the random extrinsics to an .npy file
        # pick a single intrinsic (or you could tile if you want variation)
        K = intrinsics_list[0]

        random_images = []
        for E_rand in random_extrs:
            random_images.append(self.render_view(K, E_rand))
        return random_images


class MultiViewPCDRenderer:
    def __init__(self, pcd_path: str, width: int = 1920, height: int = 1080):
        """
        Initialize the multi-view renderer for a colored point cloud.

        Args:
            pcd_path (str): Path to the point cloud file (e.g. PLY, PCD).
            width (int): Image width.
            height (int): Image height.
        """
        self.width = width
        self.height = height
        
        # Load the colored point cloud.
        self.point_cloud = o3d.io.read_point_cloud(pcd_path)
        if len(self.point_cloud.points) == 0:
            raise ValueError(f"Failed to load or empty point cloud from {pcd_path}")

        # Ensure the point cloud has color information.
        if not self.point_cloud.has_colors():
            # If not, assign a uniform color (white)
            print("no color here")
            self.point_cloud.paint_uniform_color([1.0, 1.0, 1.0])
        
        # Set up the offscreen renderer once.
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        
        # Configure the material.
        self.material = o3d.visualization.rendering.MaterialRecord()
        # Use 'defaultUnlit' so that the colors of the point cloud are preserved.
        self.material.shader = "defaultUnlit"
        self.material.shader = "defaultLit"  # Change to "defaultUnlit" if desired
        
        self.material.aspect_ratio = 1.0
        # Set the point size to a smaller value.
        self.material.point_size = 1.0  # Adjust this value (e.g., 1.0, 2.0) as needed
        
        # Add the point cloud to the renderer's scene.
        self.renderer.scene.add_geometry("point_cloud", self.point_cloud, self.material)
        # self.renderer.scene.scene.set_sun_light([1, 1, 1], [1.0, 1.0, 1.0], 300000)
        # self.renderer.scene.scene.enable_sun_light(True)        
        
        # Set a background color if desired (e.g., black).
        self.renderer.scene.set_background([0, 0, 0, 1])

    def render_view(self, intrinsic_matrix: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
        """
        Render one view given an intrinsic and extrinsic matrix.

        Args:
            intrinsic_matrix (np.ndarray): A 3x3 intrinsic matrix.
            extrinsic (np.ndarray): A 4x4 extrinsic matrix.

        Returns:
            np.ndarray: The rendered image in BGR format.
        """
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]
        pinhole_intrinsic = o3d.camera.PinholeCameraIntrinsic(self.width, self.height, fx, fy, cx, cy)
        
        # Update the camera parameters for this view.
        self.renderer.setup_camera(pinhole_intrinsic, extrinsic)
        
        # Render the view and convert the result to a NumPy array.
        rendered_image = self.renderer.render_to_image()
        rendered_np = np.asarray(rendered_image)
        
        # Convert from RGB (Open3D) to BGR (for OpenCV).
        rendered_np_bgr = cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR)
        return rendered_np_bgr

    def render_views(self, intrinsics_list: list, extrinsics_list: list, cam_idx_list: list = None) -> list:
        """
        Render and save a list of views for the given intrinsic and extrinsic matrices.

        Args:
            intrinsics_list (list): A list of 3x3 intrinsic matrices.
            extrinsics_list (list): A list of 4x4 extrinsic matrices.
            cam_idx_list (list): Indices of the views to render. If None, renders all.

        Returns:
            list: A list of rendered images in BGR format.
        """
        if cam_idx_list is None:
            cam_idx_list = list(range(len(intrinsics_list)))
            
        rendered_images = []
        for idx in cam_idx_list:
            intrinsic_matrix = intrinsics_list[idx]
            extrinsic = extrinsics_list[idx]
            
            rendered_img = self.render_view(intrinsic_matrix, extrinsic)
            # cv2.imwrite(f"rendered_pcd_{idx}.png", rendered_img)
            # print(f"Rendered image saved as rendered_pcd_{idx}.png")
            rendered_images.append(rendered_img)
            
        return rendered_images

    def render_novel_views(self, intrinsics_list: list, extrinsics_list: list, alpha: float = 0.5) -> list:
        """
        Create novel views by interpolating between each consecutive pair of provided views.
        For n provided views, n novel views are generated by interpolating between view i and view (i+1) mod n.

        Args:
            intrinsics_list (list): List of 3x3 intrinsic matrices.
            extrinsics_list (list): List of 4x4 extrinsic matrices.
            alpha (float): Interpolation factor (default is 0.5 for midpoints).

        Returns:
            list: A list of novel rendered images in BGR format.
        """
        n = len(intrinsics_list)
        novel_images = []
        for i in range(n):
            # Interpolate cyclically between view i and view (i+1)
            j = (i + 1) % n
            new_intrinsic = interpolate_intrinsics(intrinsics_list[i], intrinsics_list[j], alpha)
            new_extrinsic = interpolate_extrinsics(extrinsics_list[i], extrinsics_list[j], alpha)
            novel_img = self.render_view(new_intrinsic, new_extrinsic)
            # cv2.imwrite(f"novel_rendered_pcd_{i}.png", novel_img)
            # print(f"Saved novel_rendered_pcd_{i}.png")
            novel_images.append(novel_img)
        return novel_images


class MultiViewMeshRendererContext(MultiViewMeshRenderer):
    def __enter__(self):
        # Return self so it can be used inside the with block.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up resources here
        # E.g., if there were a cleanup method, call it
        # Otherwise, explicitly delete resources
        del self
        # Optionally force garbage collection
        import gc
        gc.collect()
        
        
class MultiViewPCDRendererContext(MultiViewPCDRenderer):
    def __enter__(self):
        # Return self so it can be used inside the with block.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up resources here
        # E.g., if there were a cleanup method, call it
        # Otherwise, explicitly delete resources
        del self
        # Optionally force garbage collection
        import gc
        gc.collect()

# Example usage:
if __name__ == '__main__':
    from option import args
    # Replace with the path to your mesh file
    mesh_path = "/home/XX/workspace/3d-measurement/texture_mesh_test/kitchen_scene1.obj"
    
    rgbd_processor = RGBD_Processor(cam_idx_list=args.cam_idx_list, extrinsic_path=args.extrinsic_path, intrinsic_path=args.intrinsic_path)
    intrinsics, _, _, extrinsics = rgbd_processor.get_calibration()    
    
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