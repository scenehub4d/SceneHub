import os, glob, argparse
import numpy as np, open3d as o3d
from tqdm import tqdm

# ─────────────────────────────────────────────────────────
#  fast_dihedral_stats
# ─────────────────────────────────────────────────────────
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
    return float(angles.mean()), float(angles.std())

# ─────────────────────────────────────────────────────────
#  mesh_metrics 
# ─────────────────────────────────────────────────────────
def mesh_metrics(mesh: o3d.geometry.TriangleMesh, unit_divisor: float = 1.0):
    verts, tris = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    n_v, n_t = len(verts), len(tris)

    a, b, c = verts[tris[:,0]], verts[tris[:,1]], verts[tris[:,2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total_area_m2 = areas.sum()       / unit_divisor**2
    mean_area_m2  = areas.mean()      / unit_divisor**2
    std_area_m2   = areas.std()       / unit_divisor**2

    mesh.compute_triangle_normals()
    mean_dh, std_dh = fast_dihedral_stats(verts, tris, np.asarray(mesh.triangle_normals))

    bbox = mesh.get_axis_aligned_bounding_box().get_extent() / unit_divisor
    return {
        "vertex_count": n_v,
        "triangle_count": n_t,
        "surface_area_m2": total_area_m2,
        "tri_area_mean_m2": mean_area_m2,
        "tri_area_std_m2": std_area_m2,
        "surface_area_mm2": total_area_m2 * 1e6,
        "tri_area_mean_mm2": mean_area_m2 * 1e6,
        "tri_area_std_mm2": std_area_m2 * 1e6,
        "dihedral_mean_deg": mean_dh,
        "dihedral_std_deg":  std_dh,
        "bbox_size_m": bbox.tolist(),
    }

# ─────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Per‑mesh metrics + summary log")
    ap.add_argument("--mesh_dir",   required=True)
    ap.add_argument("--scene_name", required=True)
    ap.add_argument("--output_dir", default="./logs")
    ap.add_argument("--unit_divisor", type=float, default=1.0)
    ap.add_argument("--max_frames",   type=int,   default=None)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.mesh_dir, "*.obj")))
    if args.max_frames: paths = paths[:args.max_frames]
    if not paths: raise RuntimeError("no *.obj found")

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, f"{args.scene_name.replace('/','_')}_mesh.log")

    per_lines, all_metrics = [], []        # --- MOD

    for p in tqdm(paths, desc="Computing meshes"):
        mesh = o3d.io.read_triangle_mesh(p)
        m = mesh_metrics(mesh, unit_divisor=args.unit_divisor)
        all_metrics.append(m)

        per_lines.append(                  # --- MOD
            f"{os.path.basename(p)}\t"
            f"{m['vertex_count']}\t{m['triangle_count']}\t"
            f"{m['surface_area_m2']:.6f}\t{m['tri_area_mean_m2']:.6f}\t"
            f"{m['tri_area_std_m2']:.6f}\t{m['surface_area_mm2']:.2f}\t"
            f"{m['tri_area_mean_mm2']:.2f}\t{m['tri_area_std_mm2']:.2f}\t"
            f"{m['dihedral_mean_deg']:.2f}\t{m['dihedral_std_deg']:.2f}\t"
            f"{m['bbox_size_m'][0]:.3f}\t{m['bbox_size_m'][1]:.3f}\t{m['bbox_size_m'][2]:.3f}"
        )

    arr = lambda k: np.array([d[k] for d in all_metrics])
    summary = {
        "Tri_Count_Max":          int(arr('triangle_count').max()),
        "Tri_Count_Mean":         float(arr('triangle_count').mean()),
        "Surf_Area_m2_Max":       float(arr('surface_area_m2').max()),
        "Surf_Area_m2_Mean":      float(arr('surface_area_m2').mean()),
        "Surf_Area_mm2_Max":      float(arr('surface_area_mm2').max()),
        "Surf_Area_mm2_Mean":     float(arr('surface_area_mm2').mean()),
        "Tri_Area_m2_Mean":       float(arr('tri_area_mean_m2').mean()),
        "Tri_Area_m2_Std":        float(arr('tri_area_std_m2').mean()),
        "Tri_Area_mm2_Mean":      float(arr('tri_area_mean_mm2').mean()),
        "Tri_Area_mm2_Std":       float(arr('tri_area_std_mm2').mean()),
        "Dihedral_Mean_deg":      float(arr('dihedral_mean_deg').mean()),
    }

    with open(log_path, "w") as f:
        # 3‑1) summary 
        f.write(
            "Scene\tTri_Count_Max\tTri_Count_Mean\t"
            "Surf_Area_m2_Max\tSurf_Area_m2_Mean\t"
            "Surf_Area_mm2_Max\tSurf_Area_mm2_Mean\t"
            "Tri_Area_m2_Mean\tTri_Area_m2_Std\t"
            "Tri_Area_mm2_Mean\tTri_Area_mm2_Std\t"
            "Dihedral_Mean_deg\n"
        )
        f.write(
            f"{args.scene_name}\t"
            f"{summary['Tri_Count_Max']}\t{summary['Tri_Count_Mean']:.1f}\t"
            f"{summary['Surf_Area_m2_Max']:.6f}\t{summary['Surf_Area_m2_Mean']:.6f}\t"
            f"{summary['Surf_Area_mm2_Max']:.2f}\t{summary['Surf_Area_mm2_Mean']:.2f}\t"
            f"{summary['Tri_Area_m2_Mean']:.6f}\t{summary['Tri_Area_m2_Std']:.6f}\t"
            f"{summary['Tri_Area_mm2_Mean']:.2f}\t{summary['Tri_Area_mm2_Std']:.2f}\t"
            f"{summary['Dihedral_Mean_deg']:.2f}\n"
        )

        # 3‑2) per‑frame 
        f.write("\n# per‑frame metrics\n")
        f.write(
            "FrameName\tVertexCount\tTriangle_Count\t"
            "Total_Surface_Area(m^2)\tMean_Triangle_Area(m^2)\tTriangle_Area_Std_Dev(m^2)\t"
            "Total_Surface_Area(mm^2)\tMean_Triangle_Area(mm^2)\tTriangle_Area_Std_Dev(mm^2)\t"
            "Mean_Dihedral_Angle(deg)\tDihedral_Angle_Std_Dev(deg)\t"
            "Bounding_Box_X_(m)\tBounding_Box_Y(m)\tBounding_Box_Z(m)\n"
        )
        f.write("\n".join(per_lines) + "\n")

    print("Metrics logged to", log_path)

if __name__ == "__main__":
    main()