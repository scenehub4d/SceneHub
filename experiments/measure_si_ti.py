import os
import glob
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def compute_si_ti_for_sequence(folder, to_gray=True, pattern="*.png", max_frames=None):
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if len(paths) < 2:
        raise RuntimeError(f"Need ≥2 frames in {folder}")
    if max_frames:
        paths = paths[:max_frames]

    si_list, ti_list = [], []
    prev = None
    for p in tqdm(paths, desc=os.path.basename(folder)):
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"failed to load {p}")
        if to_gray and img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        si_list.append(float(np.std(np.hypot(gx, gy))))
        if prev is not None:
            ti_list.append(float(np.std(cv2.absdiff(img, prev))))
        prev = img

    return {
        "si_max": max(si_list),
        "si_mean": float(np.mean(si_list)),
        "ti_max": max(ti_list),
        "ti_mean": float(np.mean(ti_list)),
    }

def main(args):
    cams = ["cam0", "cam1", "cam2", "cam3"]
    rgb_stats_all, depth_stats_all = [], []

    os.makedirs(args.output_dir, exist_ok=True)
    scene_name = os.path.basename(args.scene_dir.rstrip('/'))
    log_path = os.path.join(args.output_dir, f"{scene_name}_2d.log")

    with open(log_path, 'w') as log_file:
        # Header for tab-delimited columns
        log_file.write("Camera\tType\tSI_max\tSI_mean\tTI_max\tTI_mean\n")

        for cam in cams:
            cam_dir   = os.path.join("../XX/rgbd_data", args.scene_dir, cam)
            rgb_dir   = os.path.join(cam_dir, "rgb")
            depth_dir = os.path.join(cam_dir, "trans_depth")

            rgb_stats   = compute_si_ti_for_sequence(rgb_dir, to_gray=True,  max_frames=args.max_frames)
            depth_stats = compute_si_ti_for_sequence(depth_dir, to_gray=False, max_frames=args.max_frames)

            rgb_stats_all.append(rgb_stats)
            depth_stats_all.append(depth_stats)

            # one row per sequence
            log_file.write(f"{cam}\tRGB\t"
                           f"{rgb_stats['si_max']:.3f}\t{rgb_stats['si_mean']:.3f}\t"
                           f"{rgb_stats['ti_max']:.3f}\t{rgb_stats['ti_mean']:.3f}\n")

            log_file.write(f"{cam}\tDepth\t"
                           f"{depth_stats['si_max']:.3f}\t{depth_stats['si_mean']:.3f}\t"
                           f"{depth_stats['ti_max']:.3f}\t{depth_stats['ti_mean']:.3f}\n")

        # Compute and write averages
        def avg_stat(stats_list, key):
            return float(np.mean([s[key] for s in stats_list]))

        avg_rgb = (
            avg_stat(rgb_stats_all, 'si_max'),
            avg_stat(rgb_stats_all, 'si_mean'),
            avg_stat(rgb_stats_all, 'ti_max'),
            avg_stat(rgb_stats_all, 'ti_mean'),
        )
        avg_depth = (
            avg_stat(depth_stats_all, 'si_max'),
            avg_stat(depth_stats_all, 'si_mean'),
            avg_stat(depth_stats_all, 'ti_max'),
            avg_stat(depth_stats_all, 'ti_mean'),
        )

        log_file.write("\n")  # blank line before summary
        log_file.write(f"Average\tRGB\t{avg_rgb[0]:.3f}\t{avg_rgb[1]:.3f}\t{avg_rgb[2]:.3f}\t{avg_rgb[3]:.3f}\n")
        log_file.write(f"Average\tDepth\t{avg_depth[0]:.3f}\t{avg_depth[1]:.3f}\t{avg_depth[2]:.3f}\t{avg_depth[3]:.3f}\n")

    print(f"Logged to: {log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute SI/TI for RGB & Depth across cams 0–3 in a scene and log tab-delimited results."
    )
    parser.add_argument("--scene_dir",  required=True,
                        help="path to scene folder (e.g. …/arena_scene0)")
    parser.add_argument("--output_dir", default="./logs",
                        help="folder to save log files")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="optional: only process first N frames of each sequence")
    args = parser.parse_args()

    main(args)