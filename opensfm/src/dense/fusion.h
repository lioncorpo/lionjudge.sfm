#pragma once

#include <unordered_set>
#include <vector>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

struct StereoFusionOptions {
  // Maximum image size in either dimension.
  int max_image_size = -1;

  // Minimum number of fused pixels to produce a point.
  int min_num_pixels = 2;

  // Maximum number of pixels to fuse into a single point.
  int max_num_pixels = 10000;

  // Maximum depth in consistency graph traversal.
  int max_traversal_depth = 100;

  // Maximum relative difference between measured and projected pixel.
  double max_reproj_error = 2.0f;

  // Maximum relative difference between measured and projected depth.
  double max_depth_error = 0.01f;

  // Maximum angular difference in degrees of normals of pixels to be fused.
  double max_normal_error = 10.0f;

  // Number of overlapping images to transitively check for fusing points.
  int check_num_images = 50;

  // Cache size in gigabytes for fusion. The fusion keeps the bitmaps, depth
  // maps, normal maps, and consistency graphs of this number of images in
  // memory. A higher value leads to less disk access and faster fusion, while
  // a lower value leads to reduced memory usage. Note that a single image can
  // consume a lot of memory, if the consistency graph is dense.
  double cache_size = 32.0;
};

class StereoFusion {
 public:

  StereoFusion();

  std::vector<std::vector<int>> GetMaxOverlappingImages(
      const size_t num_images, int count) const;

  void AddView(const double *pK, const double *pR, const double *pt,
               const float *pdepth, const float *pplane,
               const unsigned char *pcolor, const unsigned char *plabel,
               const unsigned char *pdetection,
               const std::map<int, int>& shared_num_points,
               int width, int height);

  void Run(std::vector<float> *merged_points,
           std::vector<float> *merged_normals,
           std::vector<unsigned char> *merged_colors,
           std::vector<unsigned char> *merged_labels,
           std::vector<unsigned char> *merged_detections);
  void Fuse(std::vector<float> *merged_points,
            std::vector<float> *merged_normals,
            std::vector<unsigned char> *merged_colors,
            std::vector<unsigned char> *merged_labels,
            std::vector<unsigned char> *merged_detections);

  const StereoFusionOptions options_;
  
  std::vector<std::map<int, int>> shared_num_points_;
  std::vector<cv::Mat> depths_;
  std::vector<cv::Mat> planes_;
  std::vector<cv::Mat> colors_;
  std::vector<cv::Mat> labels_;
  std::vector<cv::Mat> detections_;
  std::vector<cv::Matx33d> Ks_;
  std::vector<cv::Matx33d> Rs_;
  std::vector<cv::Vec3d> ts_;

  const float max_squared_reproj_error_;
  const float min_cos_normal_error_;

  std::vector<char> used_images_;
  std::vector<char> fused_images_;
  std::vector<std::vector<int>> overlapping_images_;
  std::vector<cv::Mat> fused_pixel_masks_;
  std::vector<std::pair<int, int>> depth_map_sizes_;

  struct FusionData {
    int image_idx = -1;
    int row = 0;
    int col = 0;
    int traversal_depth = -1;
    bool operator()(const FusionData& data1, const FusionData& data2) {
      return data1.image_idx > data2.image_idx;
    }
  };

  // Next points to fuse.
  std::vector<FusionData> fusion_queue_;

  // Already fused points.
  std::vector<Eigen::Vector3d> fused_points_;
  std::vector<std::vector<int>> fused_points_visibility_;

  // Points of different pixels of the currently point to be fused.
  std::vector<float> fused_point_x_;
  std::vector<float> fused_point_y_;
  std::vector<float> fused_point_z_;
  std::vector<float> fused_point_nx_;
  std::vector<float> fused_point_ny_;
  std::vector<float> fused_point_nz_;
  std::vector<uint8_t> fused_point_r_;
  std::vector<uint8_t> fused_point_g_;
  std::vector<uint8_t> fused_point_b_;
};