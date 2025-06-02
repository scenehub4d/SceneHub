cmake_minimum_required(VERSION 3.13)
project(mesh-texturing LANGUAGES CXX)

# ————————————————————————————————
# Point CMake at your Open3D install
# ————————————————————————————————
set(OPEN3D_ROOT /home/jaehong/built_packages/open3d_install_0.19.0)
list(APPEND CMAKE_PREFIX_PATH ${OPEN3D_ROOT})

# ————————————————————————————————
# C++ settings
# ————————————————————————————————
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_policy(SET CMP0076 NEW)
cmake_policy(SET CMP0079 NEW)

# ————————————————————————————————
# Find dependencies
# ————————————————————————————————
find_package(Open3D 0.19.0 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui calib3d)
find_package(jsoncpp REQUIRED)

# ————————————————————————————————
# Include directories
# ————————————————————————————————
include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)

# ————————————————————————————————
# Build Reconstruction library (from src/reconstruction)
# ————————————————————————————————
# add_subdirectory(src/reconstruction)

# ————————————————————————————————
# Build TextureMapping library
# ————————————————————————————————
add_library(texturemapping STATIC
  src/reconstruction/texture_mapping.cpp
)

target_include_directories(texturemapping PUBLIC
  ${CMAKE_SOURCE_DIR}/include/reconstruction
)

target_link_libraries(texturemapping PUBLIC
  Open3D::Open3D
  ${OpenCV_LIBS}
  jsoncpp_lib
)

# ————————————————————————————————
# Build main executable
# ————————————————————————————————
add_executable(texture_mapper
  src/main_texture_mapping.cpp
)

add_executable(reconstruct_texture_mesh
  src/reconstruct_texture_mesh.cpp
)


target_link_libraries(texture_mapper PRIVATE
#   reconstruction
  texturemapping
)

target_link_libraries(reconstruct_texture_mesh PRIVATE
#   reconstruction
  texturemapping
)