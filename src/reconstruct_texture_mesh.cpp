#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <chrono>

#include <json/json.h>
#include <opencv2/opencv.hpp>

#include "open3d/Open3D.h"
#include "texture_mapping.hpp"  // optimized_multi_cam_uv()

using namespace open3d;
using core::Tensor;
using t::geometry::TriangleMesh;

// Single definition of calibration data
struct CalibData {
    Tensor intrinsic;     // 3×3 Float64
    Tensor extrinsic_tf;  // 4×4 Float64
};

struct Config {
    float voxel_size = 0.005f;
    int block_count = 10000;
    float depth_max = 10.0f;
    float trunc_voxel_multiplier = 8.0f;
    float depth_scale = 1000.0f;
};

inline const core::Device CPU_DEVICE{"CPU:0"};
inline const core::Device GPU_DEVICE{"CUDA:0"};

CalibData LoadCameraCalib(const std::string &intrinsic_path,
                          const std::string &extrinsic_path,
                          int deviceIdx) {
    std::ifstream ifs(intrinsic_path);
    if (!ifs) throw std::runtime_error("Cannot open intrinsics: " + intrinsic_path);
    Json::Value root;
    if (!(ifs >> root)) throw std::runtime_error("Failed to parse JSON: " + intrinsic_path);

    auto camMap = root["camera_config"];
    if (!camMap.isObject()) throw std::runtime_error("'camera_config' missing or invalid");

    std::string idxKey = std::to_string(deviceIdx);
    if (!camMap.isMember(idxKey)) throw std::runtime_error("No camera_config for index " + idxKey);

    std::string serial = camMap[idxKey].asString();
    auto intr = root["device_calibration"][serial]["optimal_intrinsics"];
    if (intr.isNull()) throw std::runtime_error("Missing optimal_intrinsics for " + serial);

    CalibData d;
    // Build intrinsic matrix in Float64
    double fx = intr["fx"].asDouble() * 0.5;
    double fy = intr["fy"].asDouble() * 0.5;
    double cx = intr["cx"].asDouble() * 0.5;
    double cy = intr["cy"].asDouble() * 0.5;
    d.intrinsic = Tensor::Empty({3,3}, core::Float64);
    double *pI = reinterpret_cast<double*>(d.intrinsic.GetDataPtr());
    pI[0]=fx; pI[1]=0;  pI[2]=cx;
    pI[3]=0;  pI[4]=fy; pI[5]=cy;
    pI[6]=0;  pI[7]=0;  pI[8]=1;

    // Load NPY extrinsics and convert to Float64
    if (!std::filesystem::exists(extrinsic_path))
        throw std::runtime_error("Extrinsics file not found: " + extrinsic_path);
    Tensor extr_all = t::io::ReadNpy(extrinsic_path);
    if (extr_all.NumDims() != 3 ||
        extr_all.GetShape()[0] <= deviceIdx ||
        extr_all.GetShape()[1] != 4 ||
        extr_all.GetShape()[2] != 4) {
        throw std::runtime_error("Invalid extrinsics shape in: " + extrinsic_path);
    }
    d.extrinsic_tf = extr_all[deviceIdx].To(core::Float64).Clone();
    return d;
}

int main(int argc, char** argv) {
    if (argc < 7 || (argc - 5) % 2 != 0) {
        std::cerr << "Usage: " << argv[0]
                  << " output_path intrinsics.json extrinsics.npy [color depth]...\n";
        return 1;
    }

    Config config;
    std::string output_path    = argv[1];
    std::string intrinsic_path = argv[2];
    std::string extrinsic_path = argv[3];
    std::string decimation_ratio = argv[4]; // 90 for 90%

    float decimation_factor = std::stof(decimation_ratio) / 100.0f;
    int device_count = (argc - 5) / 2;

    // Load calibration
    std::vector<CalibData> calib(device_count);
    for (int i = 0; i < device_count; ++i) {
        calib[i] = LoadCameraCalib(intrinsic_path, extrinsic_path, i);
    }

    // Load color and depth frames
    struct Frame { cv::Mat depth; Tensor depthGpu; };
    std::vector<Frame> frames(device_count);
    std::vector<cv::Mat> color_imgs(device_count);
    int arg = 5;
    for (int i = 0; i < device_count; ++i) {
        // Read color into color_imgs and depth into frames
        color_imgs[i] = cv::imread(argv[arg++], cv::IMREAD_COLOR);
        cv::cvtColor(color_imgs[i], color_imgs[i], cv::COLOR_BGR2RGB);
        frames[i].depth = cv::imread(argv[arg++], cv::IMREAD_UNCHANGED);
        if (color_imgs[i].empty() || frames[i].depth.empty()) {
            std::cerr << "ERR: Failed to load images for camera " << i << std::endl;
            return 1;
        }
        // Convert depth to GPU tensor
        auto& D = frames[i].depth;
        auto ptr = D.ptr<uint16_t>();
        std::vector<uint16_t> raw(ptr, ptr + D.rows * D.cols);
        Tensor depth_t(raw, {D.rows, D.cols}, core::UInt16, CPU_DEVICE);
        depth_t = depth_t.To(core::Float32)
                         .Reshape({D.rows, D.cols, 1})
                         .To(GPU_DEVICE);
        frames[i].depthGpu = depth_t;
    }

    // TSDF integration
    t::geometry::VoxelBlockGrid voxel_grid(
        {"tsdf", "weight"}, {core::Float32, core::Float32}, {{1}, {1}},
        config.voxel_size, 16, config.block_count, GPU_DEVICE);
    for (int i = 0; i < device_count; ++i) {
        auto coords = voxel_grid.GetUniqueBlockCoordinates(
            frames[i].depthGpu,
            calib[i].intrinsic,
            calib[i].extrinsic_tf,
            config.depth_scale,
            config.depth_max,
            config.trunc_voxel_multiplier);
        voxel_grid.Integrate(
            coords, frames[i].depthGpu,
            calib[i].intrinsic, calib[i].extrinsic_tf,
            config.depth_scale, config.depth_max, config.trunc_voxel_multiplier);
    }
    auto mesh = voxel_grid.ExtractTriangleMesh(0, -1);
    mesh = mesh.To(CPU_DEVICE);
    mesh.RemoveVertexAttr("normals");

    if (decimation_factor > 0) {
        // mesh = mesh.SimplifyQuadricDecimation(decimation_factor); // for some reason tensor decimation is not working

        auto legacy = mesh.ToLegacy();  // open3d::geometry::TriangleMesh
        int target = static_cast<int>(legacy.triangles_.size() * (1-decimation_factor));
        auto decimated_ptr = legacy.SimplifyQuadricDecimation(target,1e-6,1.0);
        mesh = open3d::t::geometry::TriangleMesh::FromLegacy(*decimated_ptr);
    }

    // Prepare intrinsics, extrinsics, depth imgs for UV
    std::vector<Tensor> intrinsics, extrinsics;
    std::vector<cv::Mat> depth_imgs;
    intrinsics.reserve(device_count);
    extrinsics.reserve(device_count);
    depth_imgs.reserve(device_count);
    for (int i = 0; i < device_count; ++i) {
        intrinsics.push_back(calib[i].intrinsic);
        extrinsics.push_back(calib[i].extrinsic_tf);
        depth_imgs.push_back(frames[i].depth);
    }

    // Bake UVs
    auto t0 = std::chrono::high_resolution_clock::now();
    optimized_multi_cam_uv(&mesh, intrinsics, extrinsics, &depth_imgs);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "UV baking took "
              << std::chrono::duration<double, std::milli>(t1-t0).count()
              << " ms\n";

    // Stitch colors
    cv::Mat stitched_image;
    cv::hconcat(color_imgs, stitched_image);
    if (stitched_image.empty()) {
        std::cerr << "ERR: Stitching produced empty image!" << std::endl;
        return 1;
    }

    // Convert to Open3D and attach
    geometry::Image o3d_img;
    o3d_img.Prepare(stitched_image.cols, stitched_image.rows,
                    stitched_image.channels(), 1);
    std::memcpy(o3d_img.data_.data(), stitched_image.data,
                stitched_image.total()*stitched_image.elemSize());
    auto legacy = mesh.ToLegacy();
    legacy.textures_.clear();
    legacy.textures_.push_back(o3d_img);

    // Save mesh and texture
    bool ok = io::WriteTriangleMesh(
        output_path, legacy,
        true, true, true, true, true, true);
    if (!ok) {
        std::cerr << "ERR: Failed to write mesh to " << output_path << std::endl;
        return 1;
    }
    // std::string tex_path = output_path + "_texture.png";
    // if (!cv::imwrite(tex_path, stitched_image)) {
    //     std::cerr << "ERR: Failed to save texture to " << tex_path << std::endl;
    //     return 1;
    // }
    std::cout << "Saved mesh: " << output_path << std::endl;
            //   << " and texture: " << tex_path << std::endl;
    return 0;
}
