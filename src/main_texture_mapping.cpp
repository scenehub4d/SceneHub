// src/main_texture_mapping.cpp

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>

#include <json/json.h>
#include <opencv2/opencv.hpp>

#include "open3d/Open3D.h"
#include "texture_mapping.hpp"  // optimized_multi_cam_uv()
#include <chrono>

using namespace open3d;
using core::Tensor;
using t::geometry::TriangleMesh;

struct CalibData {
    Tensor intrinsic;     // 3×3 float32
    Tensor extrinsic_tf;  // 4×4 float32
};

CalibData LoadCameraCalib(const std::string &intrinsic_path,
                          const std::string &extrinsic_path,
                          int deviceIdx) {
    // — Load JSON intrinsics
    std::ifstream ifs(intrinsic_path);
    if (!ifs)
        throw std::runtime_error("Cannot open intrinsics: " + intrinsic_path);
    Json::Value root;
    if (!(ifs >> root))
        throw std::runtime_error("Failed to parse JSON: " + intrinsic_path);

    auto camMap = root["camera_config"];
    if (!camMap.isObject())
        throw std::runtime_error("'camera_config' missing or invalid");

    std::string idxKey = std::to_string(deviceIdx);
    if (!camMap.isMember(idxKey))
        throw std::runtime_error("No camera_config for index " + idxKey);

    std::string serial = camMap[idxKey].asString();
    auto intr = root["device_calibration"][serial]["optimal_intrinsics"];
    if (intr.isNull())
        throw std::runtime_error("Missing optimal_intrinsics for " + serial);

    double fx = intr["fx"].asDouble() * 0.5;
    double fy = intr["fy"].asDouble() * 0.5;
    double cx = intr["cx"].asDouble() * 0.5;
    double cy = intr["cy"].asDouble() * 0.5;

    // Build 3×3 tensor (Float32)
    CalibData d;
    d.intrinsic = Tensor::Empty({3,3}, core::Float32);
    float *pI = reinterpret_cast<float*>(d.intrinsic.GetDataPtr());
    pI[0]=fx; pI[1]=0;  pI[2]=cx;
    pI[3]=0;  pI[4]=fy; pI[5]=cy;
    pI[6]=0;  pI[7]=0;  pI[8]=1;

    // — Load NPY extrinsics
    if (!std::filesystem::exists(extrinsic_path))
        throw std::runtime_error("Extrinsics file not found: " + extrinsic_path);
    Tensor extr_all = t::io::ReadNpy(extrinsic_path);
    if (extr_all.NumDims()!=3 ||
        extr_all.GetShape()[0] <= deviceIdx ||
        extr_all.GetShape()[1]!=4 || extr_all.GetShape()[2]!=4)
    {
        throw std::runtime_error("Invalid extrinsics shape in: " + extrinsic_path);
    }
    // slice out one 4×4 and cast to Float32
    d.extrinsic_tf = extr_all[deviceIdx]
                        .To(core::Float32)
                        .Clone();

    return d;
}

int main(int argc, char** argv) {
    if (argc != 13) {
        std::cerr << "ERR: invalid number of arguments" << argc << "\n";
        std::cerr << "Usage:\n  " << argv[0]
                  << " mesh.ply output_path intrinsics.json extrinsics.npy"
                     " color0 depth0 color1 depth1"
                     " color2 depth2 color3 depth3\n";
        return 1;
    }

    const int device_count = 4;
    int arg = 1;
    std::string mesh_path      = argv[arg++];
    std::string output_path    = argv[arg++];    
    std::string intrinsic_path = argv[arg++];
    std::string extrinsic_path = argv[arg++];

    // 1) Load mesh from file
    geometry::TriangleMesh legacy;
    if (!io::ReadTriangleMesh(mesh_path, legacy)) {
        std::cerr << "ERR: cannot read mesh: " << mesh_path << "\n";
        return 1;
    }
    TriangleMesh mesh = TriangleMesh::FromLegacy(legacy)
                            .To(core::Device("CPU:0"));

    // 2) Load calib for each camera
    std::vector<CalibData> calib(device_count);
    for (int i = 0; i < device_count; ++i)
        calib[i] = LoadCameraCalib(intrinsic_path, extrinsic_path, i);

    // 3) Load RGB + depth
    std::vector<cv::Mat> color_imgs(device_count), depth_imgs(device_count);
    for (int i = 0; i < device_count; ++i) {
        color_imgs[i] = cv::imread(argv[arg++], cv::IMREAD_COLOR);
        cv::cvtColor(color_imgs[i], color_imgs[i], cv::COLOR_BGR2RGB);
        depth_imgs[i] = cv::imread(argv[arg++], cv::IMREAD_UNCHANGED);
        if (color_imgs[i].empty() || depth_imgs[i].empty()) {
            std::cerr << "ERR: failed to load images for cam " << i << "\n";
            return 1;
        }
    }

    // 4) Extract Tensors
    std::vector<Tensor> intrinsics(device_count), extrinsics(device_count);
    for (int i = 0; i < device_count; ++i) {
        intrinsics[i] = calib[i].intrinsic;
        extrinsics[i] = calib[i].extrinsic_tf;
    }

    // 5) Bake UVs
    mesh.RemoveVertexAttr("normals");
    auto t0 = std::chrono::high_resolution_clock::now();
    optimized_multi_cam_uv(&mesh, intrinsics, extrinsics, &depth_imgs);
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "UV baking took " << elapsed_ms << " ms\n";

    // — CV Mat으로 stitched image 만들기
    std::vector<cv::Mat> cvColImgs(device_count);
    for (int i = 0; i < device_count; ++i) {
        // color_imgs는 std::vector<cv::Mat>으로 이미 읽어둔 컬러 이미지들
        cvColImgs[i] = std::move(color_imgs[i]);
    }
    cv::Mat stitched_image;
    cv::hconcat(cvColImgs, stitched_image);  // cvColImgs[0] | cvColImgs[1] | ...
    // cv::flip(stitched_image, stitched_image, /*flipCode=*/1);    

    // 7) Convert stitched cv::Mat → Open3D Image
    open3d::geometry::Image o3d_img;
    // rows  → width,    cols  → height
    o3d_img.Prepare(
        /*rows=*/stitched_image.cols,  // ← this is the width
        /*cols=*/stitched_image.rows,  // ← this is the height
        /*num_of_channels=*/stitched_image.channels(),
        /*bytes_per_channel=*/1
    );
    size_t byte_count = stitched_image.total() * stitched_image.elemSize();
    std::memcpy(o3d_img.data_.data(), stitched_image.data, byte_count);

    // 8) Attach the texture to the mesh (legacy API)
    legacy = mesh.ToLegacy();
    legacy.textures_.clear();
    legacy.textures_.push_back(o3d_img);

    // 9) Save per‐frame mesh (.obj/.mtl) and stitched image (.png)
    // std::string mesh_filename  = "mesh_frame.obj";  
    // std::string image_filename = "stitched_image_frame.png"; 

    bool ok_mesh = open3d::io::WriteTriangleMesh(
        output_path,
        legacy,
        /*write_ascii=*/true,
        /*compressed=*/true,
        /*write_vertex_normals=*/true,
        /*write_vertex_colors=*/true,
        /*write_triangle_uvs=*/true,
        /*write_materials=*/true
    );
    if (!ok_mesh) {
        std::cerr << "ERR: Failed to write mesh to " 
                  << output_path << std::endl;
        return 1;
    }

    // if (!cv::imwrite(image_filename, stitched_image)) {
    //     std::cerr << "ERR: Failed to save image to " 
    //               << image_filename << std::endl;
    //     return 1;
    // }

    std::cout << "Saved frame "<< std::endl;
    return 0;
}
