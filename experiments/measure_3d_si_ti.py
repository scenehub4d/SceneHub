import os
import glob
import argparse

import numpy as np
import open3d as o3d

from tqdm import tqdm
from typing import Optional
from sklearn.neighbors import NearestNeighbors       # fast C impl.


# ────────────────────────────────────────────────
#  Geometry-SI  = σ of local curvature distances
# ────────────────────────────────────────────────
def curvature_variance(pcd: o3d.geometry.PointCloud, k: int = 30) -> float:
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=k))
    dists = np.asarray(pcd.compute_nearest_neighbor_distance())
    return float(np.std(dists))

def geometry_si(pcd: o3d.geometry.PointCloud,
                k: int = 20,
                voxel_size: Optional[float] = None) -> float:  # ← changed
    """
    Geometry‑SI (Spatial Information) for a point cloud.
    Returns the std‑dev of local surface variation (curvature).
    """
    if voxel_size is not None:                       # ← keep explicit None‑check
        pcd = pcd.voxel_down_sample(voxel_size)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))

    pts = np.asarray(pcd.points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    curvatures = []
    for p in pts:
        _, idxs, _ = kdtree.search_knn_vector_3d(p, k)
        neigh = pts[idxs]
        centred = neigh - neigh.mean(axis=0)
        cov = centred.T @ centred / (k - 1)
        λ = np.linalg.eigvalsh(cov)          # λ₁ ≤ λ₂ ≤ λ₃
        # print(λ)
        curvatures.append(λ[0] / λ.sum())    # surface‑variation

    return float(np.std(curvatures))

# ────────────────────────────────────────────────
#  Geometry-TI  = σ of point displacements via ICP (to convergence)
# ────────────────────────────────────────────────
def ti_std_after_icp(src: o3d.geometry.PointCloud,
                     dst: o3d.geometry.PointCloud,
                     max_corr_dist: float = 0.05,
                     max_iter: int = 50) -> float:
    icp_result = o3d.pipelines.registration.registration_icp(
        src, dst, max_corr_dist,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    )
    # apply transformation
    src_aligned = src.transform(icp_result.transformation)

    corr = np.asarray(icp_result.correspondence_set)
    if corr.size == 0:
        return 0.0

    src_pts = np.asarray(src_aligned.points)[corr[:, 0]]
    dst_pts = np.asarray(dst.points)[corr[:, 1]]
    displacements = np.linalg.norm(src_pts - dst_pts, axis=1)
    return float(np.std(displacements))


# ────────────────────────────────────────────────
#  Coverage ratio  (occupied / total voxels)
# ────────────────────────────────────────────────
def coverage_ratio(pcd: o3d.geometry.PointCloud, voxel: float) -> float:
    vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel)
    occ = len(vg.get_voxels())
    extent = pcd.get_axis_aligned_bounding_box().get_extent()
    total = np.prod(np.ceil(extent / voxel))
    return float(occ) / (total + 1e-9)


# ────────────────────────────────────────────────
#  Scene size  (union AABB of the whole sequence)
# ────────────────────────────────────────────────
def sequence_bbox_size(frames, unit_divisor: float = 1.0):
    mn = np.array([ np.inf,  np.inf,  np.inf])
    mx = np.array([-np.inf, -np.inf, -np.inf])
    for f in frames:
        pcd = o3d.io.read_point_cloud(f)
        # pcd = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)[0]        
        if pcd.is_empty():
            continue  # 또는 raise error
        aabb = pcd.get_axis_aligned_bounding_box()
        min_bound = np.asarray(aabb.min_bound)
        max_bound = np.asarray(aabb.max_bound)
        mn = np.minimum(mn, min_bound)
        mx = np.maximum(mx, max_bound)
        # print(f"frame: {f}, min: {min_bound}, max: {max_bound}")
    return (mx - mn) / unit_divisor  # returns (dx, dy, dz)


# ────────────────────────────────────────────────
#  Main metrics extractor
# ────────────────────────────────────────────────
def sequence_metrics(seq_dir: str,
                     voxel_size: float = 0.01,
                     max_corr_dist: float = 0.05,
                     unit_divisor: float = 1000.0,
                     unit: str = "m",
                     max_frames: int = None):
    frames = sorted(glob.glob(os.path.join(seq_dir, "*.ply")))
    if len(frames) < 2:
        raise RuntimeError("Need ≥2 frames for TI.")
    if max_frames:
        frames = frames[:max_frames]

    si_g, ti_g, npts, cover = [], [], [], []
    prev = None
    per_frame = [] 
    for idx, f in enumerate(tqdm(frames, desc=os.path.basename(seq_dir))):
        p = o3d.io.read_point_cloud(f)
        if unit == "mm":
            conversion_factor= 1000.0 #100.0  # convert to meters
        elif unit == "cm":
            conversion_factor = 100.0
        elif unit == "m":
            conversion_factor = 1.0
        p.points = o3d.utility.Vector3dVector(np.asarray(p.points) / conversion_factor)
        # Print the first xx points
        # xx = 10  # Replace with the desired number of points to display
        # print(f"First {xx} points from {f}:")
        # print(np.asarray(p.points)[:xx])
        npts.append(len(p.points))
        # si_g.append(curvature_variance(p))
        curvature_variance(p) #to estimate normal
        cover.append(coverage_ratio(p, voxel_size))

        p_ds = p.voxel_down_sample(voxel_size)
        if prev is not None:
            ti_g.append(ti_std_after_icp(prev, p_ds, max_corr_dist))
        prev = p_ds
        # si_val = geometry_si(p_ds)       # ← 잠깐 변수에 담음
        si_val = 0
        si_g.append(si_val)   

        per_frame.append({
            "frame": os.path.basename(f),
            "si":    si_val,
            "ti":    ti_g[-1] if idx else 0.0,   # 첫 프레임은 TI=0
            "npts":  npts[-1],
            "cover": cover[-1]
        })      

    bbox = sequence_bbox_size(frames, unit_divisor=1.0)
    print(f"bbox: {bbox}")
    return {
        "V_SI_G_max":   max(si_g),
        "V_TI_G_max":   max(ti_g),
        "point_mean":   float(np.mean(npts)),
        "point_std":    float(np.std(npts)),
        "coverage_mean": float(np.mean(cover)),
        "coverage_std":  float(np.std(cover)),
        "scene_size_m":  tuple(bbox),
        "per_frame":    per_frame  # ← 새 항목
    }


# ──────────────────────────────────────────────────────────────
#  CLI driver with tab-delimited logging to <scene>_3d.log
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Compute 3D geometry metrics and log tab-delimited results."
    )
    parser.add_argument("--scene_dir",
                        help="path to point-cloud sequence directory (e.g. …/arena_scene0/gt/color_ptcl)")       
    parser.add_argument("--scene_name",
                        help="name of the scene")
    parser.add_argument("--output_dir", default="./logs",
                        help="where to write the log file")
    parser.add_argument("--voxel_size",   type=float, default=0.01,
                        help="voxel size for coverage and down-sampling")
    parser.add_argument("--max_corr_dist", type=float, default=0.05,
                        help="max correspondence distance for ICP")
    parser.add_argument("--unit_divisor", type=float, default=1000.0,
                        help="divide bounding-box dimensions by this to get scene_size units")
    parser.add_argument("--unit", type=str, choices=["mm", "cm", "m"], default="m",
                        help="Unit for bounding-box dimensions (options: mm, cm, m). Default is meters (m).")    
    parser.add_argument("--max_frames",   type=int, default=None,
                        help="optional: only process first N frames")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # scene_name = os.path.basename(args.scene_dir.rstrip("/"))
    # scene_name = args.scene_name 
    args.scene_name = args.scene_name.replace("/", "_")
    log_path = os.path.join(args.output_dir, f"{args.scene_name}_3d.log")

    # args.scene_dir = os.path.join("../output", args.scene_name, "gt/color_ptcl")
    
    # compute metrics
    res = sequence_metrics(
        args.scene_dir,
        voxel_size=args.voxel_size,
        max_corr_dist=args.max_corr_dist,
        unit_divisor=args.unit_divisor,
        unit=args.unit,
        max_frames=args.max_frames
    )

    # write tab-delimited log
    with open(log_path, 'w') as log_file:
        log_file.write(
            "Scene\tV_SI_G_max_mm\tV_TI_G_max_mm\t"
            "point_mean\tpoint_std\tcoverage_mean\tcoverage_std\t"
            "scene_size_x_m\tscene_size_y_m\tscene_size_z_m\n"
        )

        v_si = res["V_SI_G_max"] #* 1000.0
        v_ti = res["V_TI_G_max"] * args.unit_divisor
        pm, ps = res["point_mean"], res["point_std"]
        cm, cs = res["coverage_mean"], res["coverage_std"]
        sx, sy, sz = res["scene_size_m"]

        log_file.write(
            f"{args.scene_name}\t"
            f"{v_si:.4f}\t{v_ti:.4f}\t"
            f"{pm:.0f}\t{ps:.0f}\t"
            f"{cm:.3f}\t{cs:.3f}\t"
            f"{sx:.2f}\t{sy:.2f}\t{sz:.2f}\n"
        )
        
        # ── per‑frame header ─────────────────────
        log_file.write(
            "\n# per‑frame metrics\n"
            "frame\tSI_G\tTI_G\tpoints\tcoverage\n"
        )
        
        # ── per‑frame rows ───────────────────────
        for rec in res["per_frame"]:
            log_file.write(
                f"{rec['frame']}\t"
                f"{rec['si']:.4f}\t{rec['ti']:.4f}\t"
                f"{rec['npts']}\t{rec['cover']:.4f}\n"
            )        
        
    print(f"3D metrics logged to: {log_path}")


if __name__ == "__main__":
    main()