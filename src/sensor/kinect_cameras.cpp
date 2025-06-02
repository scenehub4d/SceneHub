#include "kinect_cameras.hpp"

#include <jsoncpp/json/json.h>
#include <signal.h>

#include <string>
#include <thread>

#include "kinect_capture.hpp"
#include "util.hpp"

#define COLOR_EXPOSURE_USEC 50000
#define POWERLINE_FREQ 2

static std::atomic<bool> stop_cameras = false;

static void sigint_handler(int)
{
  stop_cameras = true;
}

// Function to generate a formatted timestamp string
std::string generateTimestamp() {
  auto now = std::chrono::system_clock::now();
  auto time = std::chrono::system_clock::to_time_t(now);
  auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

  std::stringstream ss;
  ss << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S") << "_" << std::setfill('0') << std::setw(3) << milliseconds.count();
  return ss.str();
}

KinectCameras::KinectCameras(
    PipeDataInCollection<ImageFrame> *out,
    PipeDataInCollectionOnce<CalibData> *calibOut,
    std::vector<std::string> *serial_nums,
    int device_count,
    std::string master_serial,
    int width,
    int height,
    Json::Value jsonConf_calibration,
    bool enable_depth)
    : out_(out), calib_data_out_(calibOut), conf_({(uint32_t)device_count, master_serial, width, height}), calibration_config_(jsonConf_calibration), enable_depth_(enable_depth)
{
  signal(SIGINT, sigint_handler);

  for (uint32_t i = 0; i < conf_.nCams; i++)
  {
    this->threads_.push_back(
        std::thread(static_cast<void (KinectCameras::*)(uint32_t, std::string *)>(&KinectCameras::run), this, i, &(serial_nums->at(i))));
  }
}

KinectCameras::KinectCameras(
    PipeDataInCollection<ImageFrame> *out,
    PipeDataInCollectionOnce<CalibData> *calibOut,
    std::vector<std::string> *serial_nums,
    int device_count,
    std::string master_serial,
    int width,
    int height,
    Json::Value jsonConf_calibration,
    bool enable_depth,
    int frame_count)
    : out_(out), calib_data_out_(calibOut), conf_({(uint32_t)device_count, master_serial, width, height}), calibration_config_(jsonConf_calibration), enable_depth_(enable_depth), frame_count_(frame_count)
{
  signal(SIGINT, sigint_handler);

  for (uint32_t i = 0; i < conf_.nCams; i++)
  {
    this->threads_.push_back(
        std::thread(static_cast<void (KinectCameras::*)(uint32_t, std::string *, int)>(&KinectCameras::run), this, i, &(serial_nums->at(i)), frame_count));
  }
}

KinectCameras::~KinectCameras()
{
  for (auto &t : this->threads_)
  {
    t.join();
  }
  std::cerr << "Cameras stopped\n";
}

static int32_t getSyncTiming(size_t idx, size_t nCams)
{
  return 160 * idx - 80 * (nCams - 1);
}

void KinectCameras::run(uint32_t idx, std::string *serial_num)
{
  std::unique_ptr<KinectCapture> cap = std::make_unique<KinectCapture>(idx, this->enable_depth_);

  std::cout << "KinectCameras::run() ";
  cap->set_resolution(this->conf_.width, this->conf_.height);

  std::cout << "set resolution: " << this->conf_.width << "x"
            << this->conf_.height << std::endl;

  if (this->enable_depth_)
  {
    if (this->conf_.nCams == 1)
    {
      std::cout << "running on single camera mode" << std::endl;
      cap->config.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE;
    }
    else
    {

      std::cout << "running on " << this->conf_.nCams << " cameras" << std::endl;

      cap->device.set_color_control(
          K4A_COLOR_CONTROL_EXPOSURE_TIME_ABSOLUTE,
          K4A_COLOR_CONTROL_MODE_MANUAL,
          COLOR_EXPOSURE_USEC);

      cap->device.set_color_control(
          K4A_COLOR_CONTROL_POWERLINE_FREQUENCY,
          K4A_COLOR_CONTROL_MODE_MANUAL,
          POWERLINE_FREQ);

      cap->device.set_color_control(
          K4A_COLOR_CONTROL_WHITEBALANCE, K4A_COLOR_CONTROL_MODE_AUTO, 0);

      if (cap->serial_num.compare(this->conf_.master) == 0)
      {
        std::cout << "master detected" << std::endl;
        cap->config.wired_sync_mode = K4A_WIRED_SYNC_MODE_MASTER;
        cap->config.subordinate_delay_off_master_usec = 0;
        cap->config.depth_delay_off_color_usec = getSyncTiming(idx, this->conf_.nCams);
      }
      else
      {
        std::cout << "not master camera" << std::endl;
        cap->config.wired_sync_mode = K4A_WIRED_SYNC_MODE_SUBORDINATE;
        cap->config.subordinate_delay_off_master_usec = 0;
        cap->config.depth_delay_off_color_usec = getSyncTiming(idx, this->conf_.nCams);
      }
    }
  }
  else
  {
    std::cout << "no depth" << std::endl;
    cap->config.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE;
  }

  // Use when calibration is needed online (not from file)
  // cap->get_factory_calibration();

  // Load calibration data (use when calibration is from a file)
  CalibData calib_data = calibration_loader(
      this->calibration_config_, idx, cap->serial_num, this->calib_data_out_);
  std::cout << "loaded calibration data: " << cap->serial_num << std::endl;
  *serial_num = cap->serial_num;

  cap->load_intrinsic_calibration(
      calib_data.eigen_intrinsic,
      calib_data.eigen_optimal_intrinsic,
      calib_data.distortion_params);
  cap->init_undistort_map(false);

  cap->device.start_cameras(&cap->config);
  std::cout << "camera " << cap->serial_num << " started" << std::endl;

  cap->capture_frame();

  // while (!stop_cameras)
  // {
  //   cap->capture_frame();

  //   open3d::t::geometry::Image o3d_color;
  //   open3d::t::geometry::Image o3d_depth;

  // //  if (this->enable_depth_)
  // //  {
  // //    o3d_color = open3d::core::Tensor(
  // //        reinterpret_cast<const uint8_t*>(cap->cv_color_img_rectified.data),
  // //        {cap->cv_color_img.rows, cap->cv_color_img.cols, 3},
  // //        open3d::core::UInt8,
  // //        CPU_DEVICE);

  // //    o3d_depth = open3d::core::Tensor(
  // //        reinterpret_cast<const uint16_t*>(cap->cv_depth_img_rectified.data),
  // //        {cap->cv_depth_img.rows, cap->cv_depth_img.cols, 1},
  // //        open3d::core::UInt16,
  // //        CPU_DEVICE);
  // //  }

  //   this->out_->put(
  //       idx,
  //       {.cv_colImg = cap->cv_color_img_rectified,
  //        .cv_depImg = cap->cv_depth_img_rectified,
  //        .o3d_colImg = o3d_color,
  //        .o3d_depImg = o3d_depth});
  // }
}

/*
 * This function is used to capture a single frame from all cameras
 * (transformed to its respective rgb perspective)
 */
void KinectCameras::run(uint32_t idx, std::string *serial_num, int frame_count)
{
  std::unique_ptr<KinectCapture> cap = std::make_unique<KinectCapture>(idx, this->enable_depth_, frame_count);

  std::cout << "KinectCameras::run_capture_only() ";
  cap->set_resolution(this->conf_.width, this->conf_.height);

  std::cout << "set resolution: " << this->conf_.width << "x"
            << this->conf_.height << std::endl;

  if (this->enable_depth_)
  {
    if (this->conf_.nCams == 1)
    {
      std::cout << "running on single camera mode" << std::endl;
      cap->config.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE;
    }
    else
    {

      std::cout << "running on " << this->conf_.nCams << " cameras" << std::endl;

      cap->device.set_color_control(
          K4A_COLOR_CONTROL_EXPOSURE_TIME_ABSOLUTE,
          K4A_COLOR_CONTROL_MODE_MANUAL,
          COLOR_EXPOSURE_USEC);

      cap->device.set_color_control(
          K4A_COLOR_CONTROL_POWERLINE_FREQUENCY,
          K4A_COLOR_CONTROL_MODE_MANUAL,
          POWERLINE_FREQ);

      cap->device.set_color_control(
          K4A_COLOR_CONTROL_WHITEBALANCE, K4A_COLOR_CONTROL_MODE_AUTO, 0);

      if (cap->serial_num.compare(this->conf_.master) == 0)
      {
        std::cout << "master detected" << std::endl;
        cap->config.wired_sync_mode = K4A_WIRED_SYNC_MODE_MASTER;
        cap->config.subordinate_delay_off_master_usec = 0;
        cap->config.depth_delay_off_color_usec = getSyncTiming(idx, this->conf_.nCams);
      }
      else
      {
        std::cout << "not master camera" << std::endl;
        cap->config.wired_sync_mode = K4A_WIRED_SYNC_MODE_SUBORDINATE;
        cap->config.subordinate_delay_off_master_usec = 0;
        cap->config.depth_delay_off_color_usec = getSyncTiming(idx, this->conf_.nCams);
      }
    }
  }
  else
  {
    std::cout << "no depth" << std::endl;
    cap->config.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE;
  }

  // Use when calibration is needed online (not from file)
  // cap->get_factory_calibration();

  // Load calibration data (use when calibration is from a file)
  CalibData calib_data = calibration_loader(
      this->calibration_config_, idx, cap->serial_num, this->calib_data_out_);
  std::cout << "loaded calibration data: " << cap->serial_num << std::endl;
  *serial_num = cap->serial_num;

  cap->load_intrinsic_calibration(
      calib_data.eigen_intrinsic,
      calib_data.eigen_optimal_intrinsic,
      calib_data.distortion_params);
  cap->init_undistort_map(false);
  
  int captured_num = 0;
  
  if (cap->count > 0)
  {
    cap->device.start_cameras(&cap->config);
    std::cout << "camera " << cap->serial_num << " started" << std::endl;
    
    while (cap->count--) {
      cap->capture_frame();
      captured_num += 1;
      open3d::t::geometry::Image o3d_color;
      open3d::t::geometry::Image o3d_depth;

      //    if (this->enable_depth_)
      //    {
      //      o3d_color = open3d::core::Tensor(
      //          reinterpret_cast<const uint8_t*>(cap->cv_color_img_rectified.data),
      //          {cap->cv_color_img.rows, cap->cv_color_img.cols, 3},
      //          open3d::core::UInt8,
      //          CPU_DEVICE);
      //
      //      o3d_depth = open3d::core::Tensor(
      //          reinterpret_cast<const uint16_t*>(cap->cv_depth_img_rectified.data),
      //          {cap->cv_depth_img.rows, cap->cv_depth_img.cols, 1},
      //          open3d::core::UInt16,
      //          CPU_DEVICE);
      //    }

      // Generate a timestamp for the frame
      auto timestamp = std::chrono::high_resolution_clock::now();
      std::string timestamp_str = generateTimestamp();

      ImageFrame frame = {
          .cv_colImg = cap->cv_color_img_rectified,
          .cv_depImg = cap->cv_depth_img_rectified,
          .o3d_colImg = o3d_color,
          .o3d_depImg = o3d_depth,
          .serial_num = cap->serial_num,
          .count = captured_num,
          .timestamp = timestamp_str // Set the timestamp
      };

      this->out_->put(
          idx,
          frame);
    }
    std::cout << "capture while loop finished" << std::endl;
  }
}

std::vector<std::thread> *KinectCameras::get_cam_threads()
{
  return &(this->threads_);
}
