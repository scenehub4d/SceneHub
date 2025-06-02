import numpy as np
import cv2
import open3d as o3d

def transform_vertices(vertices, extrinsic, invert=True):
    """
    Transform vertices from world space to camera coordinates.
    
    Parameters:
      vertices: (N, 3) NumPy array.
      extrinsic: (4, 4) transformation matrix.
      invert: If True, the extrinsic matrix is inverted before use.
      
    Returns:
      Transformed vertices as (N, 3) array.
    """
    # if invert:
    #     extrinsic = np.linalg.inv(extrinsic)
    N = vertices.shape[0]
    vertices_h = np.hstack((vertices, np.ones((N, 1))))
    transformed = (extrinsic @ vertices_h.T).T
    # Homogeneous division:
    transformed = transformed[:, :3] / transformed[:, 3:4]
    return transformed

def vertices_to_uv(vertices_cam, intrinsic):
    """
    Project vertices (in camera coordinates) into image (pixel) coordinates.
    
    Parameters:
      vertices_cam: (N, 3) NumPy array.
      intrinsic: (3, 3) NumPy array.
    
    Returns:
      (N, 2) array of [u, v] pixel coordinates.
    """
    X = vertices_cam[:, 0]
    Y = vertices_cam[:, 1]
    Z = vertices_cam[:, 2]
    # Avoid division by zero by replacing very small Z values with a small constant.
    Z_safe = np.where(np.abs(Z) < 1e-6, 1e-6, Z)
    u = (intrinsic[0, 0] * X / Z_safe) + intrinsic[0, 2]
    v = (intrinsic[1, 1] * Y / Z_safe) + intrinsic[1, 2]
    return np.stack((u, v), axis=1)

def multi_occlusion_test(uv, vertices_cam, depth_img):
    """
    For each vertex, compute an occlusion error by comparing the projected
    depth (converted to millimeters) with the depth image.
    
    Parameters:
      uv: (N, 2) array of projected pixel coordinates.
      vertices_cam: (N, 3) vertices in camera space (Z in meters).
      depth_img: Depth image (numpy array, uint16, in millimeters).
    
    Returns:
      (N,) numpy array of absolute depth differences.
    """
    N = vertices_cam.shape[0]
    height, width = depth_img.shape[:2]
    z_error = np.zeros(N)
    for i in range(N):
        u, v = uv[i]
        u_int = int(round(u))
        v_int = int(round(v))
        if u_int < 0 or u_int >= width or v_int < 0 or v_int >= height:
            z_error[i] = 1e6  # Assign a very high error if out-of-bounds.
        else:
            depth_val = float(depth_img[v_int, u_int])
            vertex_z_mm = vertices_cam[i, 2] * 1000.0  # Convert meters to millimeters.
            z_error[i] = abs(vertex_z_mm - depth_val)
    return z_error

def compute_uv_mapping(mesh, intrinsic_list, extrinsic_list, depth_img_list, max_threshold=50):
    """
    For every vertex in the mesh, choose the best camera based on occlusion error
    using a threshold and compute the UV coordinate for a texture atlas.
    
    Parameters:
      mesh: open3d.geometry.TriangleMesh (legacy format).
      intrinsic_list: List of (3, 3) NumPy arrays (one per camera).
      extrinsic_list: List of (4, 4) NumPy arrays (one per camera).
      depth_img_list: List of depth images (NumPy arrays, uint16).
      max_threshold: Maximum acceptable occlusion error (default is 50).
    
    Process:
      1. For each camera, transform vertices and compute their UV coordinates.
      2. For each vertex, store the occlusion error (difference between projected depth
         and actual depth).
      3. For each vertex, if a camera’s error is below max_threshold and the current
         candidate’s error is above max_threshold, select the new camera.
      4. For each triangle, assign the UV coordinates using the best camera per vertex.
         Shift the U coordinate by the camera index and normalize by the atlas dimensions.
    
    Returns:
      The mesh with updated triangle_uvs.
    """
    # Get vertices and triangle indices.
    vertices = np.asarray(mesh.vertices)  # Shape (N, 3)
    triangles = np.asarray(mesh.triangles)  # Shape (M, 3)
    N = vertices.shape[0]
    device_count = len(intrinsic_list)
    
    # Assume all cameras have the same resolution (from the first depth image).
    H_RES = depth_img_list[0].shape[1]
    V_RES = depth_img_list[0].shape[0]
    
    # For each camera, compute projected UVs and occlusion errors.
    all_uvs = np.zeros((device_count, N, 2))
    all_z_errors = np.zeros((device_count, N))
    for i in range(device_count):
        intrinsic = intrinsic_list[i]
        extrinsic = extrinsic_list[i]
        depth_img = depth_img_list[i]
        vertices_cam = transform_vertices(vertices, extrinsic)
        uv = vertices_to_uv(vertices_cam, intrinsic)
        z_error = multi_occlusion_test(uv, vertices_cam, depth_img)
        all_uvs[i] = uv
        all_z_errors[i] = z_error
    
    # Choose the best camera for each vertex based on the max_threshold logic.
    best_cam_indices = np.zeros(N, dtype=int)
    for i in range(N):
        candidate = 0  # Start with camera 0.
        candidate_error = all_z_errors[0, i]
        for j in range(1, device_count):
            # If camera j's error is below the threshold and candidate's error is above,
            # update the candidate.
            if all_z_errors[j, i] < max_threshold and candidate_error > max_threshold:
                candidate = j
                candidate_error = all_z_errors[j, i]
        best_cam_indices[i] = candidate
    
    # For each triangle, assign UVs using the best camera for each vertex.
    triangle_uvs = []
    for tri in triangles:
        for idx in tri:
            cam_idx = best_cam_indices[idx]
            uv = all_uvs[cam_idx, idx].copy()
            # Shift the U coordinate by the camera index to place it in the correct atlas slot.
            uv[0] += H_RES * cam_idx
            # Normalize the UV coordinates:
            u_norm = uv[0] / (H_RES * device_count)
            v_norm = uv[1] / V_RES  # Flip V coordinate (Open3D expects origin at bottom-left).
            triangle_uvs.append([u_norm, v_norm])
    
    mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)
    return mesh

def compute_color_mapping(pc, intrinsic_list, extrinsic_list, depth_img_list, rgb_img_list, max_threshold=50):
    """
    For every point in the point cloud (pc), choose the best camera based on occlusion error
    computed from the depth images and map its color from the corresponding RGB image.
    
    Parameters:
      pc             : open3d.geometry.PointCloud (legacy format) with points in world coordinates.
      intrinsic_list : List of (3, 3) NumPy arrays (one per camera).
      extrinsic_list : List of (4, 4) NumPy arrays (one per camera).
      depth_img_list : List of depth images (NumPy arrays, uint16, in millimeters) for occlusion testing.
      rgb_img_list   : List of RGB images (NumPy arrays, H x W x 3, in RGB order) to sample colors.
      max_threshold  : Maximum acceptable occlusion error (default is 50).
    
    Returns:
      pc             : The point cloud with updated colors.
    """
    # Get point cloud vertices (points in world coordinates)
    points = np.asarray(pc.points)
    N = points.shape[0]
    device_count = len(intrinsic_list)
    
    # Assume all depth (and RGB) images share the same resolution.
    H_RES = depth_img_list[0].shape[1]
    V_RES = depth_img_list[0].shape[0]
    
    # For each camera, compute the projected UVs and occlusion errors.
    all_uvs = np.zeros((device_count, N, 2))
    all_z_errors = np.zeros((device_count, N))
    
    for i in range(device_count):
        intrinsic = intrinsic_list[i]
        extrinsic = extrinsic_list[i]
        depth_img = depth_img_list[i]
        # Transform points to camera coordinates.
        points_cam = transform_vertices(points, extrinsic)
        # Project points into image space.
        uv = vertices_to_uv(points_cam, intrinsic)
        # Compute occlusion error.
        z_error = multi_occlusion_test(uv, points_cam, depth_img)
        all_uvs[i] = uv
        all_z_errors[i] = z_error
    
    # For each point, choose the best camera based on occlusion error.
    best_cam_indices = np.zeros(N, dtype=int)
    for i in range(N):
        candidate = 0  # start with camera 0
        candidate_error = all_z_errors[0, i]
        for j in range(1, device_count):
            if all_z_errors[j, i] < max_threshold and candidate_error > max_threshold:
                candidate = j
                candidate_error = all_z_errors[j, i]
        best_cam_indices[i] = candidate
    
    # For each point, sample the RGB color from the chosen camera.
    colors = np.zeros((N, 3))
    for i in range(N):
        cam_idx = best_cam_indices[i]
        uv = all_uvs[cam_idx, i]
        u_int = int(round(uv[0]))
        v_int = int(round(uv[1]))
        if u_int < 0 or u_int >= H_RES or v_int < 0 or v_int >= V_RES:
            colors[i] = [0, 0, 0]
        else:
            colors[i] = rgb_img_list[cam_idx][v_int, u_int, :].astype(np.float64) / 255.0

    pc.colors = o3d.utility.Vector3dVector(colors)
    return pc

def optimized_multi_cam_uv(mesh, intrinsic_list, extrinsic_list, depth_img_list, max_threshold=50):
    """
    Wrapper that applies multi-view UV mapping with threshold logic.
    """
    return compute_uv_mapping(mesh, intrinsic_list, extrinsic_list, depth_img_list, max_threshold)