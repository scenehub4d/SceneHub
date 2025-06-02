from rgbd_processor import RGBD_Processor
from multiview_renderer import *
from skimage.metrics import structural_similarity as ssim
import os
from draco import Draco
import shutil
import re
import pymeshlab as ml


def calc_ssim(image1, image2):
    # Convert images from BGR (OpenCV default) to grayscale.
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Optionally, resize images to a common shape if they differ.
    # For example:
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

    # ---------------------------
    # 3. Compute the SSIM
    # ---------------------------
    # The 'full' parameter returns both the SSIM value and the difference image.
    ssim_value, diff = ssim(gray1, gray2, full=True, data_range=255.0)
    # print("SSIM between the two images:", ssim_value)

    return ssim_value

def calc_ssim_no_bg(img1, img2, bg_thresh=0):
    # 1) to gray
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # 2) resize if needed
    if g1.shape != g2.shape:
        g2 = cv2.resize(g2, (g1.shape[1], g1.shape[0]))
    # 3) compute full SSIM map
    _ , ssim_map = ssim(g1, g2, full=True, data_range=255)
    # 4) build mask of “foreground” (i.e. both pixels > threshold)
    mask = (g1 > bg_thresh) & (g2 > bg_thresh)
    # 5) average only over mask
    return float(ssim_map[mask].mean())

def fast_dihedral_stats(verts: np.ndarray, tris: np.ndarray, normals: np.ndarray,
                        coplanar_tol: float = 1e-3):
    tris_idx = np.repeat(np.arange(tris.shape[0]), 3)
    edges = np.vstack([tris[:, [0,1]], tris[:, [1,2]], tris[:, [2,0]]])
    edges = np.sort(edges, 1)
    order = np.lexsort((edges[:,1], edges[:,0]))
    edges_s, tris_idx_s = edges[order], tris_idx[order]
    same = np.all(edges_s[1:] == edges_s[:-1], 1)
    f0, f1 = tris_idx_s[:-1][same], tris_idx_s[1:][same]

    cosang = np.einsum("ij,ij->i", normals[f0], normals[f1]).clip(-1.0, 1.0)
    keep = np.abs(cosang) < 1.0 - coplanar_tol
    if not np.any(keep):
        return 0.0, 0.0
    angles = np.degrees(np.arccos(cosang[keep]))
    return float(angles.mean()), float(angles.std()), float(angles.max()), float(angles.min())

def compute_mesh_stats(mesh_path, unit_divisor=1.0):
    """
    Reads a triangle mesh from mesh_path and returns:
      n_vertices, n_triangles, total_area, mean_area, std_area
    Areas are divided by unit_divisor**2 (e.g. 1.0 for meters, 1e-3 for mm).
    """
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    verts = np.asarray(mesh.vertices)
    tris  = np.asarray(mesh.triangles)
    n_v, n_t = len(verts), len(tris)

    # per-triangle areas
    a = verts[tris[:,0]]
    b = verts[tris[:,1]]
    c = verts[tris[:,2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

    total_area = areas.sum()       / unit_divisor**2
    mean_area  = areas.mean()      / unit_divisor**2
    std_area   = areas.std()       / unit_divisor**2

    mesh.compute_triangle_normals()
    mean_dh, std_dh, max_dh, min_dh = fast_dihedral_stats(verts, tris, np.asarray(mesh.triangle_normals))

    return {
        'n_v':        n_v,
        'n_t':        n_t,
        'total_area': total_area,
        'mean_area':  mean_area,
        'std_area':   std_area,
        'mean_dh':    mean_dh,
        'std_dh':     std_dh,
        'max_dh':     max_dh,
        'min_dh':     min_dh,
    }

def test_multiview_ssim(args, filepath, save_path):
    print(args.extrinsic_path)
    rgbd_processor = RGBD_Processor(cam_idx_list=args.cam_idx_list, extrinsic_path=args.extrinsic_path, intrinsic_path=args.intrinsic_path)
    intrinsics, _, _, extrinsics = rgbd_processor.get_calibration()     
        
    filesize = os.path.getsize(filepath)
    num_sample = 50
    random_extrinsics = np.load(f"../camera_config/{args.scene_type}_random_extrinsics_seed42_sample{num_sample}.npy")
    
    with MultiViewMeshRendererContext(filepath, random_extrinsics=random_extrinsics) as renderer:
        original_render_imgs = renderer.render_views(intrinsics, extrinsics, args.cam_idx_list)

        os.makedirs(f"{save_path}/original", exist_ok=True)
        os.makedirs(f"{save_path}/interpolated", exist_ok=True)
        os.makedirs(f"{save_path}/shifted", exist_ok=True)
        os.makedirs(f"{save_path}/random", exist_ok=True)

        idx = 0
        for img in original_render_imgs:
            cv2.imwrite(f"{save_path}/original/rendered_{idx}.png", img)
            idx += 1

        interpolated_render_imgs = renderer.render_interpolated_views(intrinsics, extrinsics)

        idx = 0
        for img in interpolated_render_imgs:
            cv2.imwrite(f"{save_path}/interpolated/interpolated_rendered_{idx}.png", img)
            idx += 1
    
        shifted_render_imgs = renderer.render_shifted_views(intrinsics, extrinsics)

        idx = 0
        for img in shifted_render_imgs:
            cv2.imwrite(f"{save_path}/shifted/shifted_rendered_{idx}.png", img)
            idx += 1

        random_render_imgs = renderer.render_random_views(intrinsics, num_samples=num_sample)            
        idx = 0
        for img in random_render_imgs:
            cv2.imwrite(f"{save_path}/random/random_rendered_{idx}.png", img)
            idx += 1            
    
    del renderer

    return original_render_imgs, shifted_render_imgs, interpolated_render_imgs, random_render_imgs, filesize


if __name__ == '__main__':
    from option import args
    
    # name = "output_decimation_arena_scene3"
    # base_path = f"/home/XX/workspace/3d-measurement/{name}/gt/textured_mesh"
    base_path = f"{args.data_path}/gt/textured_mesh"
    gt_file = f"{base_path}/0.obj"

    if args.pcvd:
        base_path = f"./pcvd_rendered/{args.scene_name}"
        gt_file = f"/home/XX/dataset/PCVD/Dataset_Folder/{args.scene_name}/3Dmesh/mesh-f00001.obj"

    os.makedirs(base_path, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    # ——— open log and write header ———
    log_path = f"{args.output_dir}/{args.scene_name}_mesh_stats.log"
    with open(log_path, "w") as log:
        log.write(
            "scene_name\t"
            "decimate\tSize\tVertices\tTriangles\t"
            "Total_Area(m^2)\tMean_Area(m^2)\tStd_Area(m^2)\t"
            "Mean_Dh\tStd_Dh\tMax_Dh\tMin_Dh\t"
            "Average_SSIM\tOriginal\tShifted\tInterpolated\tRandom\n"
        )

        # ——— compute GT metrics ———
        gt_original, gt_shifted, gt_interpolated, gt_random, filesize = \
            test_multiview_ssim(args, gt_file, f"{base_path}/gt")

        # mesh stats via function
        mesh_stats = compute_mesh_stats(gt_file)

        # ——— write GT row (SSIM=1 by definition) ———
        log.write(
            f"{args.scene_name}\t"
            f"gt\t"
            f"{filesize}\t"
            f"{mesh_stats['n_v']}\t"
            f"{mesh_stats['n_t']}\t"
            f"{mesh_stats['total_area']:.6f}\t"
            f"{mesh_stats['mean_area']:.8f}\t"
            f"{mesh_stats['std_area']:.8f}\t"
            f"{mesh_stats['mean_dh']:.2f}\t"
            f"{mesh_stats['std_dh']:.2f}\t"
            f"{mesh_stats['max_dh']:.2f}\t"
            f"{mesh_stats['min_dh']:.2f}\t"
            f"{1}\t"
            f"{1}\t"
            f"{1}\t"
            f"{1}\t"
            f"{1}\n"
        )
        
        if args.pcvd:
            exit(1)
        # ——— loop over decimations ———
        # decimation_values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        decimation_values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        for dv in decimation_values:
            save_path = f"{base_path}/decimated_{dv}"
            os.makedirs(save_path, exist_ok=True)

            dv_file = f"{base_path}_decimated_{dv}/0.obj"
            (dec_o, dec_s, dec_i, dec_r, _) = \
                test_multiview_ssim(args, dv_file, save_path)
            filesize = os.path.getsize(dv_file)

            # collect SSIM lists
            orig_ssim = [calc_ssim_no_bg(g, d) for g,d in zip(gt_original,   dec_o)]
            shft_ssim = [calc_ssim_no_bg(g, d) for g,d in zip(gt_shifted,    dec_s)]
            intp_ssim = [calc_ssim_no_bg(g, d) for g,d in zip(gt_interpolated,dec_i)]
            rndm_ssim = [calc_ssim_no_bg(g, d) for g,d in zip(gt_random,     dec_r)]
            all_ssim  = orig_ssim + shft_ssim + intp_ssim + rndm_ssim

            # mesh stats via function
            mesh_stats = compute_mesh_stats(dv_file)

            # write one line for this decimation
            log.write(
                f"{args.scene_name}\t"
                f"{dv}\t"
                f"{filesize}\t"
                f"{mesh_stats['n_v']}\t"
                f"{mesh_stats['n_t']}\t"
                f"{mesh_stats['total_area']:.6f}\t"
                f"{mesh_stats['mean_area']:.8f}\t"
                f"{mesh_stats['std_area']:.8f}\t"
                f"{mesh_stats['mean_dh']:.2f}\t"
                f"{mesh_stats['std_dh']:.2f}\t"
                f"{mesh_stats['max_dh']:.2f}\t"
                f"{mesh_stats['min_dh']:.2f}\t"
                f"{np.mean(all_ssim):.9f}\t"
                f"{np.mean(orig_ssim):.9f}\t"
                f"{np.mean(shft_ssim):.9f}\t"
                f"{np.mean(intp_ssim):.9f}\t"
                f"{np.mean(rndm_ssim):.9f}\n"
            )
    print(f"Written full table to {log_path}")
     
# if __name__ == '__main__':
#     name = "output_decimation_arena_scene3"
#     base_path = f"/home/XX/workspace/3d-measurement/{name}/gt/textured_mesh"
#     gt_file = f"{base_path}/0.obj"

#     save_path = f"./gt"
#     gt_original, gt_shifted, gt_interpolated, gt_random, filesize = test_multiview_ssim(args, gt_file, save_path)
#     print(f"GT filesize size(bytes): {filesize}")

#     mesh = o3d.io.read_triangle_mesh(gt_file)
#     verts, tris = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
#     n_v, n_t = len(verts), len(tris)
#     unit_divisor = 1.0
#     a, b, c = verts[tris[:,0]], verts[tris[:,1]], verts[tris[:,2]]
#     areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
#     total_area_m2 = areas.sum()       / unit_divisor**2
#     mean_area_m2  = areas.mean()      / unit_divisor**2
#     std_area_m2   = areas.std()       / unit_divisor**2    
    
#     print(f"Mesh Statistics for gt:")
#     print("Unit\tVertices\tTriangles\tTotal Area\tMean Area\tStd Area")
#     print(f"m²\t{n_v}\t{n_t}\t{total_area_m2:.6f}\t{mean_area_m2:.8f}\t{std_area_m2:.8f}")
#     print(f"mm²\t\t\t{total_area_m2 * 1e6:.2f}\t{mean_area_m2 * 1e6:.4f}\t{std_area_m2 * 1e6:.4f}")    
    
    
#     decimation_values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

#     for dv in decimation_values:
#         save_path = f"./decimated_{dv}"
#         os.makedirs(save_path, exist_ok=True)        
    
    
    
#         dv_file = f"{base_path}_decimated_{dv}/0.obj"    
#         dec_original, dec_shifted, dec_interpolated, dec_random, filesize = test_multiview_ssim(args, dv_file, save_path)
#         filesize = os.path.getsize(dv_file)
#         ssim_values = []
        
#         original_ssim_values = []
#         shifted_ssim_values = []
#         interpolated_ssim_values = []
#         random_ssim_values = []
        
#         for gt, dec in zip(gt_original, dec_original):
#             ssim_value = calc_ssim_no_bg(gt, dec)
#             ssim_values.append(ssim_value)
#             original_ssim_values.append(ssim_value)
#         print(f"[original] SSIM between GT and Decimated image: {np.mean(original_ssim_values)}")

#         for gt, dec in zip(gt_shifted, dec_shifted):
#             ssim_value = calc_ssim_no_bg(gt, dec)
#             ssim_values.append(ssim_value)
#             shifted_ssim_values.append(ssim_value)
#         print(f"[shifted] SSIM between GT and Decimated image: {np.mean(shifted_ssim_values)}")        
        
#         for gt, dec in zip(gt_interpolated, dec_interpolated):
#             ssim_value = calc_ssim_no_bg(gt, dec)
#             ssim_values.append(ssim_value)
#             interpolated_ssim_values.append(ssim_value)
#         print(f"[interpolated] SSIM between GT and Decimated image: {np.mean(interpolated_ssim_values)}")                    

#         idx = 0
#         for gt, dec in zip(gt_random, dec_random):
#             ssim_value = calc_ssim_no_bg(gt, dec)
#             # print(f"idx: {idx}, ssim_value: {ssim_value}")
#             ssim_values.append(ssim_value)
#             random_ssim_values.append(ssim_value)
#         print(f"[random] SSIM between GT and Decimated image: {np.mean(random_ssim_values)}")                
        
#         mesh = o3d.io.read_triangle_mesh(dv_file)
        

#         verts, tris = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
#         n_v, n_t = len(verts), len(tris)
#         unit_divisor = 1.0
#         a, b, c = verts[tris[:,0]], verts[tris[:,1]], verts[tris[:,2]]
#         areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
#         total_area_m2 = areas.sum()       / unit_divisor**2
#         mean_area_m2  = areas.mean()      / unit_divisor**2
#         std_area_m2   = areas.std()       / unit_divisor**2            
        
        
#         print(f"Mesh Statistics for {dv}:")
#         print("Unit\tVertices\tTriangles\tTotal Area\tMean Area\tStd Area")
#         print(f"m²\t{n_v}\t{n_t}\t{total_area_m2:.6f}\t{mean_area_m2:.8f}\t{std_area_m2:.8f}")
#         print(f"mm²\t\t\t{total_area_m2 * 1e6:.2f}\t{mean_area_m2 * 1e6:.4f}\t{std_area_m2 * 1e6:.4f}")
        
#         print(f"Average SSIM for {dv}: {np.mean(ssim_values)}, size(bytes): {filesize}")