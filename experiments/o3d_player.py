import open3d
import glob
import re
import numpy as np
import argparse

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def visualize(path, num_frames=600):
    # Gather all the frame-wise point-cloud files (excluding camera/f32 files)
    pcds = [
        f for f in glob.glob(f'{path}/*.ply')
        if not any(cam in f for cam in ["cam0", "cam1", "cam2", "cam3", "f32"])
    ]
    pcds = sorted(pcds, key=natural_sort_key)
    if len(pcds) == 0:
        print("No point-cloud files found in:", path)
        return

    # Open3D visualizer setup
    vis = open3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=3840, height=2160)
    render_option = vis.get_render_option()
    render_option.point_size = 1

    # Load the first dynamic point cloud
    idx = 0
    current_pcd = open3d.io.read_point_cloud(pcds[idx])

    # Hardcode the mesh background path exactly as in the original script:
    hardcoded_background = "/XX/MM_release/photogrammetry/kitchen/kitchen.obj"
    background_geom = None
    # try:
    #     background_geom = open3d.io.read_point_cloud(hardcoded_background)
    #     if len(background_geom.points) == 0:
    #         background_geom = open3d.io.read_triangle_mesh(
    #             hardcoded_background, enable_post_processing=True
    #         )
    # except:
    #     background_geom = open3d.io.read_triangle_mesh(
    #         hardcoded_background, enable_post_processing=True
    #     )

    if background_geom is None:
        print("Failed to load hardcoded background from:", hardcoded_background)
    else:
        vis.add_geometry(background_geom)

    # Add the initial point cloud on top
    vis.add_geometry(current_pcd)
    pcd_handle = current_pcd  # Keep a reference so we can remove/update it later

    # === Initial camera pose adjustment ===
    view_control = vis.get_view_control()
    camera_params = view_control.convert_to_pinhole_camera_parameters()

    # Elevate camera (translate along Y axis)
    # T_up = np.eye(4)
    # T_up[1, 3] += 0.5
    # # Tilt downward by ~15 degrees
    # pitch_rad = np.radians(-15.0)
    # cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
    # R_pitch = np.array([
    #     [1,      0,       0, 0],
    #     [0,  cos_p, -sin_p, 0],
    #     [0,  sin_p,  cos_p, 0],
    #     [0,      0,      0, 1]
    # ])
    # camera_params.extrinsic = R_pitch @ T_up @ camera_params.extrinsic
    # view_control.convert_from_pinhole_camera_parameters(camera_params)

    def update_point_cloud(vis):
        nonlocal idx, pcd_handle, current_pcd

        # Advance frame index
        idx = (idx + 1) % len(pcds)

        # Save current camera parameters so we can restore them
        view_ctrl = vis.get_view_control()
        cam_params = view_ctrl.convert_to_pinhole_camera_parameters()

        # Remove only the previous point cloud
        vis.remove_geometry(pcd_handle, reset_bounding_box=False)

        # Load the next frame's point cloud
        new_pcd = open3d.io.read_point_cloud(pcds[idx])
        current_pcd = new_pcd
        pcd_handle = new_pcd

        # Add the new point cloud on top of the existing background
        vis.add_geometry(pcd_handle, reset_bounding_box=False)

        # Restore camera
        view_ctrl.convert_from_pinhole_camera_parameters(cam_params)

        # Apply a small yaw rotation each frame (rotating the camera around Y)
        angle_rad = np.radians(1.0)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        R_yaw = np.array([
            [ cos_a, 0, sin_a, 0],
            [     0, 1,     0, 0],
            [-sin_a, 0, cos_a, 0],
            [     0, 0,     0, 1]
        ])
        cam_params.extrinsic = cam_params.extrinsic @ R_yaw
        view_ctrl.convert_from_pinhole_camera_parameters(cam_params)

        return False  # keep the callback alive

    def right_click(vis):
        nonlocal idx, pcd_handle, current_pcd

        # On right-click, advance to next frame similarly (no continuous rotation)
        idx = (idx + 1) % len(pcds)
        view_ctrl = vis.get_view_control()
        cam_params = view_ctrl.convert_to_pinhole_camera_parameters()

        vis.remove_geometry(pcd_handle, reset_bounding_box=False)
        new_pcd = open3d.io.read_point_cloud(pcds[idx])
        current_pcd = new_pcd
        pcd_handle = new_pcd
        vis.add_geometry(pcd_handle, reset_bounding_box=False)
        view_ctrl.convert_from_pinhole_camera_parameters(cam_params)
        return False

    def print_view_matrix(vis):
        view_ctrl = vis.get_view_control()
        cam_params = view_ctrl.convert_to_pinhole_camera_parameters()
        print("Extrinsic (view) matrix:")
        print(cam_params.extrinsic)
        print("Intrinsic matrix:")
        print(cam_params.intrinsic.intrinsic_matrix)
        np.savetxt("view_matrix.txt", cam_params.extrinsic)

    def exit_key(vis):
        vis.destroy_window()

    def capture_view(vis):
        vis.capture_screen_image("captured_view.png")
        print("Captured view saved as captured_view.png")

    # Register callbacks
    vis.register_key_callback(ord("C"), capture_view)
    vis.register_key_callback(262, right_click)  # Right arrow key
    vis.register_key_callback(32, exit_key)      # Space bar
    vis.register_key_callback(ord("P"), print_view_matrix)
    vis.register_animation_callback(update_point_cloud)

    vis.run()
    vis.destroy_window()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="o3d player")
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help="Directory containing the frame-wise .ply files"
    )
    args = parser.parse_args()

    visualize(args.data_path)
