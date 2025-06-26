import numpy as np
import open3d as o3d
from multiview_renderer import *
from argparse import ArgumentParser

# map your semantic names → RGB triples
COLOR_MAP = {
    "original":     [1.0, 0.0, 0.0],  # red
    "random":       [0.0, 1.0, 0.0],  # green
    "interpolated": [0.0, 0.0, 1.0],  # blue
    "shifted":      [0.0, 0.0, 0.0],  # magenta   
    # # 3) Prepare a small palette of RGB colors
    # palette = [
    #     [1.0, 0.0, 0.0],  # red
    #     [0.0, 1.0, 0.0],  # green
    #     [0.0, 0.0, 1.0],  # blue
    #     [1.0, 1.0, 0.0],  # yellow
    #     [1.0, 0.0, 1.0],  # magenta
    #     [0.0, 1.0, 1.0],  # cyan
    # ]

}


def make_colored_frusta(extrinsics, intrinsic_matrix,
                        width, height, scale,
                        cam_type: str) -> list[o3d.geometry.LineSet]:
    """
    Build and color a batch of tiny camera frusta.
      - extrinsics: list/array of 4×4 poses
      - cam_type: must be a key in COLOR_MAP
    """
    color = COLOR_MAP[cam_type]
    frusta = []
    for E in extrinsics:
        fr = o3d.geometry.LineSet.create_camera_visualization(
            width, height, intrinsic_matrix, E
        )
        # compute world-space camera center
        cam_center = np.linalg.inv(E)[:3, 3]
        fr.scale(scale, center=cam_center)

        # apply uniform color
        n_lines = len(fr.lines)
        fr.colors = o3d.utility.Vector3dVector(np.tile(color, (n_lines, 1)))
        frusta.append(fr)
    return frusta

def camera_center_from_extrinsic(E, convention="w2c"):
    """
    Return camera center (x, y, z) in *world* coordinates.
    convention = "w2c"  →  E maps world → camera   (Open3D, COLMAP default)
    convention = "c2w"  →  E maps camera → world   (Blender-style)
    """
    R = E[:3, :3]
    t = E[:3, 3]

    if convention == "w2c":      # world → cam
        return -(R.T @ t)        # C = -Rᵀ t # C = np.linalg.inv(E)[:3, 3]
    elif convention == "c2w":    # cam → world
        return t                 # C = translation
    else:
        raise ValueError("convention must be 'w2c' or 'c2w'")

def camera_direction_from_extrinsic(E, convention="w2c"):
    R = E[:3, :3]
    if convention == "w2c":
        return (R.T @ np.array([0, 0, 1]))  # camera -Z axis in world
    elif convention == "c2w":
        return R @ np.array([0, 0, -1])      # same
    else:
        raise ValueError("Invalid convention")
    
def create_camera_frustum(E, convention="w2c", size=0.2):
    """
    Given a single extrinsic matrix E, return a triangle mesh representing
    a simple camera frustum (pyramid) in world coordinates.
    """
    C = camera_center_from_extrinsic(E, convention)
    forward = camera_direction_from_extrinsic(E, convention)
    up = np.array([0, 1, 0])

    right = np.cross(forward, up)
    up = np.cross(right, forward)

    forward = forward / np.linalg.norm(forward)
    up = up / np.linalg.norm(up)
    right = right / np.linalg.norm(right)

    # four corners of the image plane
    scale = size * 0.5
    img_plane_center = C + forward * size
    corners = [
        img_plane_center + scale * (up + right),
        img_plane_center + scale * (up - right),
        img_plane_center + scale * (-up - right),
        img_plane_center + scale * (-up + right),
    ]

    vertices = [C] + corners
    triangles = [
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 1],
        [1, 2, 3],
        [1, 3, 4],
    ]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    mesh.paint_uniform_color([0.1, 0.1, 1.0])
    mesh.compute_vertex_normals()
    return mesh

def spheres_from_extrinsics(extrinsics, radius=0.05, color=[1.0, 0.0, 0.0]):
    spheres = []
    for E in extrinsics:
        C = camera_center_from_extrinsic(E)
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sph.compute_vertex_normals()
        sph.paint_uniform_color(color)
        sph.translate(C)
        spheres.append(sph)
    return spheres


if __name__ == "__main__":
    parser = ArgumentParser()           
    parser.add_argument('--scene', type=str, default="arena")
    parser.add_argument('--scene_name', type=str, default="scene1")
    parser.add_argument(
        '--add_background',
        action='store_true',
        help='if set, include background geometry in the visualization'
    )
        
    args=parser.parse_args()   
        
    data_name = f"{args.scene}_{args.scene_name}"
    # mesh_path = f"/home/XX/workspace/3d-measurement/output_for_alignment_photogram_aligned/{data_name}/gt/textured_mesh/0.obj"
    mesh_path = f"/home/XX/workspace/3d-measurement/final_alignment/{data_name}/gt/textured_mesh/0.obj"    
    # background_path = f"/XX/XX/scaniverse_scan/{args.scene}/{args.scene}.obj"
    background_path = f"/XX/XX/photogram_scan/{args.scene}/{args.scene}.obj"    
    # extrinsics_path = f"/XX/XX/extrinsics/global_extrinsics_{args.scene}_scaniverse_aligned.npy"
    extrinsics_path = f"/XX/XX/extrinsics/global_extrinsics_{args.scene}_photogram_aligned.npy"

    # 2) load all extrinsics
    extrinsics = np.load(extrinsics_path)

    # Window size for frustum drawing (must match your intrinsics)
    WIDTH, HEIGHT = 1280, 720
    SCALE = 0.2    # 10× smaller    
    
    # If you have a saved intrinsic, load it; otherwise you can define one:
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=WIDTH, height=HEIGHT,
        fx=525.0, fy=525.0,
        cx=640.0, cy=360.0
    )
    intrinsic_matrix = intrinsic.intrinsic_matrix


    # ——— Load data ———
    mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=True)
    if args.add_background:
        mesh_background = o3d.io.read_triangle_mesh(background_path, enable_post_processing=True)
        # tmesh = o3d.t.io.read_triangle_mesh(mesh_path)
        # # tmesh.textures will now contain your images
        # mesh = tmesh.to_legacy()
        mesh_background.compute_vertex_normals()
    mesh.compute_vertex_normals()
    

    # # 3) Cluster triangles into connected components
    # #    tri_labels: length = #triangles, giving a component ID per triangle
    # #    counts:     array giving size (#tris) of each component
    # tri_labels, counts, _ = mesh.cluster_connected_triangles()
    # tri_labels = np.asarray(tri_labels)
    # counts     = np.asarray(counts)

    # # 4) Remove all components smaller than some threshold
    # min_triangles = 50
    # small_ids = np.where(counts < min_triangles)[0]
    # # build list of triangle indices to delete
    # to_delete = [i for i, lbl in enumerate(tri_labels) if lbl in small_ids]
    # mesh.remove_triangles_by_index(to_delete)
    # mesh.remove_unreferenced_vertices()


    # 3) turn them into world-space camera centers
    centers = [camera_center_from_extrinsic(E, convention="w2c")
               for E in extrinsics]
    centers = np.vstack(centers)
    
    # 4) for each center, make a small red sphere
    spheres = []
    for C in centers:
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.06)
        sph.compute_vertex_normals()
        sph.paint_uniform_color([1.0, 0.0, 0.0])
        sph.translate(C)
        spheres.append(sph)
    
    # # 5) merge mesh + all spheres into one big mesh
    # combined = mesh
    # for sph in spheres:
    #     combined += sph


    original_cameras = extrinsics.copy()
    shifted_cameras = []
    interpolated_cameras = []
    
    random_seed = 42
    num_samples = 25
    num_samples = 50
    random_cameras = sample_camera_extrinsics_inside_bbox_mesh(mesh, num_samples=num_samples, random_seed=random_seed, original_ext=np.linalg.inv(original_cameras))

    np.save(f"../camera_config/{args.scene}_random_extrinsics_seed{random_seed}_sample{num_samples}.npy", np.array(random_cameras))
            
    n = len(extrinsics)
    for i in range(n):
        # Interpolate cyclically between view i and view (i+1)
        j = (i + 1) % n
        # new_intrinsic = interpolate_intrinsics(intrinsics_list[i], intrinsics_list[j], alpha)
        new_extrinsic = interpolate_extrinsics(extrinsics[i], extrinsics[j], 0.5)
        interpolated_cameras.append(new_extrinsic)

    
    directions = ["up", "down", "left", "right"]
    amount = 0.2 #0.1  # adjust as needed

    for ext in extrinsics:
        for dir in directions:
            shifted = shift_extrinsic(ext, dir, amount=amount)
            shifted_cameras.append(shifted)

    original_frusta = make_colored_frusta(original_cameras, intrinsic_matrix, WIDTH, HEIGHT, SCALE, "original")
    random_frusta = make_colored_frusta(random_cameras, intrinsic_matrix, WIDTH, HEIGHT, SCALE, "random")
    inter_frusta = make_colored_frusta(interpolated_cameras, intrinsic_matrix, WIDTH, HEIGHT, SCALE, "interpolated")
    shifted_frusta = make_colored_frusta(shifted_cameras, intrinsic_matrix, WIDTH, HEIGHT, SCALE, "shifted")

    # save original, interpolated, and shifted views too
    np.save(f"../camera_config/{args.scene}_original.npy",
            np.array(original_cameras))

    np.save(f"../camera_config/{args.scene}_interpolated.npy",
            np.array(interpolated_cameras))

    np.save(f"../camera_config/{args.scene}_shifted.npy",
            np.array(shifted_cameras))


    # spheres_original = spheres_from_extrinsics(original_cameras, radius=0.06, color=[1.0, 0.0, 0.0])
    # spheres_random = spheres_from_extrinsics(random_cameras, radius=0.05, color=[0.0, 1.0, 0.0])
    # spheres_interp = spheres_from_extrinsics(interpolated_cameras, radius=0.05, color=[0.0, 0.0, 1.0])
    # spheres_shifted = spheres_from_extrinsics(shifted_cameras, radius=0.04, color=[0.0, 0.0, 0.0])

    # spheres = spheres_original + spheres_random + spheres_interp + spheres_shifted    

    # collect all and draw
    all_frusta = original_frusta + random_frusta + inter_frusta + shifted_frusta    

    if args.add_background:
        mesh_data = [mesh, mesh_background, *spheres, *all_frusta]
    else:
        mesh_data = [mesh, *spheres, *all_frusta]
    
    # # ——— Visualize all together ———
    # o3d.visualization.draw_geometries(
    #     mesh_data,
    #     window_name="Mesh + Camera Poses",
    #     width=WIDTH, height=HEIGHT, 
    # )    
    
    # assume mesh_data is a list of your geometries
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Mesh + Camera Poses",
        width=WIDTH,
        height=HEIGHT
    )
    for g in mesh_data:
        vis.add_geometry(g)
        
    opt = vis.get_render_option()
    # opt.light_on = False   # turn OFF Phong lighting
    opt.light_on = True  # turn it back ON

    vis.run()
    vis.destroy_window()        
    
    exit (1)
    
    positions = sample_camera_extrinsics_inside_bbox_mesh(mesh, num_samples=40)
    extrinsics = np.array(positions)
    
    # 3) turn them into world-space camera centers
    centers = [camera_center_from_extrinsic(E, convention="w2c")
               for E in extrinsics]
    centers = np.vstack(centers)
    
    # 4) for each center, make a small red sphere
    spheres = []
    for C in centers:
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        sph.compute_vertex_normals()
        sph.paint_uniform_color([1.0, 0.0, 0.0])
        sph.translate(C)
        spheres.append(sph)
    
    # 5) merge mesh + all spheres into one big mesh
    combined = mesh
    for sph in spheres:
        combined += sph
    
    # 6) save out as one .ply (will carry per‐vertex colors if your mesh had any)
    o3d.io.write_triangle_mesh("mesh_with_cameras.ply", combined, write_ascii=True)
    print("Wrote mesh_with_cameras.ply:", len(combined.triangles), "triangles total.")