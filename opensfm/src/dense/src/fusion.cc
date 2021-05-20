// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include <dense/fusion.h>
#include <iostream>


template <typename T>
float Median(std::vector<T>* elems) {
  const size_t mid_idx = elems->size() / 2;
  std::nth_element(elems->begin(), elems->begin() + mid_idx, elems->end());
  if (elems->size() % 2 == 0) {
    const float mid_element1 = static_cast<float>((*elems)[mid_idx]);
    const float mid_element2 = static_cast<float>(
        *std::max_element(elems->begin(), elems->begin() + mid_idx));
    return (mid_element1 + mid_element2) / 2.0f;
  } else {
    return static_cast<float>((*elems)[mid_idx]);
  }
}

// Use the sparse model to find most connected image that has not yet been
// fused. This is used as a heuristic to ensure that the workspace cache reuses
// already cached images as efficient as possible.
int FindNextImage(const std::vector<std::vector<int>>& overlapping_images,
                  const std::vector<char>& used_images,
                  const std::vector<char>& fused_images,
                  const int prev_image_idx) {
  for (const auto image_idx : overlapping_images.at(prev_image_idx)) {
    if (used_images.at(image_idx) && !fused_images.at(image_idx)) {
      return image_idx;
    }
  }

  // If none of the overlapping images are not yet fused, simply return the
  // first image that has not yet been fused.
  for (size_t image_idx = 0; image_idx < fused_images.size(); ++image_idx) {
    if (used_images[image_idx] && !fused_images[image_idx]) {
      return image_idx;
    }
  }

  return -1;
}

float DegToRad(const float deg) {
  return deg * 0.0174532925199432954743716805978692718781530857086181640625f;
}

StereoFusion::StereoFusion()
    : options_(StereoFusionOptions()),
      max_squared_reproj_error_(options_.max_reproj_error *
                                options_.max_reproj_error),
      min_cos_normal_error_(std::cos(DegToRad(options_.max_normal_error))) {
}

cv::Vec3d Project(const cv::Vec3d &x,
                  const cv::Matx33d &K,
                  const cv::Matx33d &R,
                  const cv::Vec3d &t) {
  return K * (R * x + t);
}

cv::Vec3d Backproject(double x, double y, double depth,
                      const cv::Matx33d &K,
                      const cv::Matx33d &R,
                      const cv::Vec3d &t) {
  return R.t() * (depth * K.inv() * cv::Vec3d(x, y, 1) - t);
}

double DegToRad(const double deg) {
  return deg * 0.0174532925199432954743716805978692718781530857086181640625;
}

std::vector<std::vector<int>> StereoFusion::GetMaxOverlappingImages(
    const size_t num_images,
    const int count) const {
  std::vector<std::vector<int>> overlapping_images(count);

  for (size_t image_idx = 0; image_idx < count; ++image_idx) {
    const auto& shared_images = shared_num_points_.at(image_idx);

    std::vector<std::pair<int, int>> ordered_images;
    ordered_images.reserve(shared_images.size());
    for (const auto& image : shared_images) {
      ordered_images.emplace_back(image.first, image.second);
    }

    const size_t eff_num_images = std::min(ordered_images.size(), num_images);
    if (eff_num_images < shared_images.size()) {
      std::partial_sort(ordered_images.begin(),
                        ordered_images.begin() + eff_num_images,
                        ordered_images.end(),
                        [](const std::pair<int, int> image1,
                           const std::pair<int, int> image2) {
                          return image1.second > image2.second;
                        });
    } else {
      std::sort(ordered_images.begin(), ordered_images.end(),
                [](const std::pair<int, int> image1,
                   const std::pair<int, int> image2) {
                  return image1.second > image2.second;
                });
    }

    overlapping_images[image_idx].reserve(eff_num_images);
    for (size_t i = 0; i < eff_num_images; ++i) {
      overlapping_images[image_idx].push_back(ordered_images[i].first);
    }
  }

  return overlapping_images;
}

void StereoFusion::AddView(const double *pK, const double *pR, const double *pt,
                           const float *pdepth, const float *pplane,
                           const unsigned char *pcolor,
                           const unsigned char *plabel,
                           const unsigned char *pdetection,
                           const std::map<int, int> &shared_num_points,
                           int width, int height) {
  Ks_.emplace_back(pK);
  Rs_.emplace_back(pR);
  ts_.emplace_back(pt);
  depths_.emplace_back(cv::Mat(height, width, CV_32F, (void *)pdepth).clone());
  planes_.emplace_back(cv::Mat(height, width, CV_32FC3, (void *)pplane).clone());
  colors_.emplace_back(cv::Mat(height, width, CV_8UC3, (void *)pcolor).clone());
  labels_.emplace_back(cv::Mat(height, width, CV_8U, (void *)plabel).clone());
  detections_.emplace_back(cv::Mat(height, width, CV_8U, (void *)pdetection).clone());
  shared_num_points_.push_back(shared_num_points);
}

void StereoFusion::Run(std::vector<float> *merged_points,
             std::vector<float> *merged_normals,
             std::vector<unsigned char> *merged_colors,
             std::vector<unsigned char> *merged_labels,
             std::vector<unsigned char> *merged_detections) {

  const auto images_count = depths_.size();
  overlapping_images_ =
      GetMaxOverlappingImages(options_.check_num_images, images_count);

  used_images_.resize(images_count, false);
  fused_images_.resize(images_count, false);
  fused_pixel_masks_.resize(images_count);
  depth_map_sizes_.resize(images_count);

  for (int i = 0; i < images_count; ++i) {

    const auto& image = colors_[i];
    const auto& depth_map = depths_[i];

    used_images_.at(i) = true;

    fused_pixel_masks_.at(i) = cv::Mat(depth_map.rows, depth_map.cols, CV_8UC1);
    fused_pixel_masks_.at(i).setTo(0);

    depth_map_sizes_.at(i) =
        std::make_pair(depth_map.cols, depth_map.rows);

    std::cout << "Computing with " << depth_map_sizes_.size() << " images" << std::endl;
  }

  size_t num_fused_images = 0;
  for (int image_idx = 0; image_idx >= 0;
       image_idx = FindNextImage(overlapping_images_, used_images_,
                                           fused_images_, image_idx)) {

    std::cout << "Processing image " << image_idx << std::endl;
    const int width = depth_map_sizes_.at(image_idx).first;
    const int height = depth_map_sizes_.at(image_idx).second;
    const auto& fused_pixel_mask = fused_pixel_masks_.at(image_idx);

    FusionData data;
    data.image_idx = image_idx;
    data.traversal_depth = 0;

    for (data.row = 0; data.row < height; ++data.row) {
      for (data.col = 0; data.col < width; ++data.col) {
        if (fused_pixel_mask.at<bool>(data.row, data.col)) {
          continue;
        }

        fusion_queue_.push_back(data);
        Fuse(merged_points, merged_normals, merged_colors, merged_labels, merged_detections);
      }
    }

    std::cout << "Fused " << merged_points->size() << " points" << std::endl;
    num_fused_images += 1;
    fused_images_.at(image_idx) = true;
  }
}

void StereoFusion::Fuse(std::vector<float>* merged_points,
                        std::vector<float>* merged_normals,
                        std::vector<unsigned char>* merged_colors,
                        std::vector<unsigned char>* merged_labels,
                        std::vector<unsigned char>* merged_detections) {
  Eigen::Vector4f fused_ref_point = Eigen::Vector4f::Zero();
  Eigen::Vector3f fused_ref_normal = Eigen::Vector3f::Zero();

  fused_point_x_.clear();
  fused_point_y_.clear();
  fused_point_z_.clear();
  fused_point_nx_.clear();
  fused_point_ny_.clear();
  fused_point_nz_.clear();
  fused_point_r_.clear();
  fused_point_g_.clear();
  fused_point_b_.clear();

  while (!fusion_queue_.empty()) {
    const auto data = fusion_queue_.back();
    const int image_idx = data.image_idx;
    const int row = data.row;
    const int col = data.col;
    const int traversal_depth = data.traversal_depth;

    fusion_queue_.pop_back();

    // Check if pixel already fused.
    auto& fused_pixel_mask = fused_pixel_masks_.at(image_idx);
    if (fused_pixel_mask.at<bool>(row, col)) {
      continue;
    }

    const auto& depth_map = depths_[image_idx];
    const float depth = depth_map.at<float>(row, col);

    // Pixels with negative depth are filtered.
    if (depth <= 0.0f) {
      continue;
    }

    // If the traversal depth is greater than zero, the initial reference
    // pixel has already been added and we need to check for consistency.
    if (traversal_depth > 0) {

      // Project reference point into current view.
      const cv::Vec3f fused_ref_point_cv(fused_ref_point(0), fused_ref_point(1), fused_ref_point(2));
      const cv::Vec3f cv_reprojection = Project(fused_ref_point_cv, Ks_[image_idx], Rs_[image_idx], ts_[image_idx]);
      const Eigen::Vector3f proj(cv_reprojection(0), cv_reprojection(1), cv_reprojection(2));

      // Depth error of reference depth with current depth.
      const float depth_error = std::abs((proj(2) - depth) / depth);
      if (depth_error > options_.max_depth_error) {
        //std::cout << "Skip depth " << depth_error << std::endl;
        continue;
      }

      // Reprojection error reference point in the current view.
      const float col_diff = proj(0) / proj(2) - col;
      const float row_diff = proj(1) / proj(2) - row;
      const float squared_reproj_error =
          col_diff * col_diff + row_diff * row_diff;
      if (squared_reproj_error > max_squared_reproj_error_) {
        //std::cout << "Skip reproj" << squared_reproj_error << std::endl;
        continue;
      }
    }

    // Determine normal direction in global reference frame.
    cv::Matx33f Rinv = Rs_[image_idx].t();
    cv::Vec3f cv_normal = Rinv*cv::normalize(planes_[image_idx].at<cv::Vec3f>(row, col));
    const Eigen::Vector3f normal(cv_normal(0), cv_normal(1), cv_normal(2));

    // Check for consistent normal direction with reference normal.
    if (traversal_depth > 0) {
      const float cos_normal_error = fused_ref_normal.dot(normal);
      if (cos_normal_error < min_cos_normal_error_) {
        //std::cout << "Skip normal" << std::endl;
        continue;
      }
    }

    // Determine 3D location of current depth value.
    const cv::Vec3f xyz_cv = Backproject(col, row, depth, Ks_[image_idx], Rs_[image_idx], ts_[image_idx]);
    const Eigen::Vector3f xyz(xyz_cv(0), xyz_cv(1), xyz_cv(2));

    // Read the color of the pixel.
    const auto& image = colors_[image_idx];
    cv::Vec3b color = image.at<cv::Vec3b>(row, col);

    // Set the current pixel as visited.
    fused_pixel_mask.at<bool>(row, col) = true;

    // Accumulate statistics for fused point.
    fused_point_x_.push_back(xyz(0));
    fused_point_y_.push_back(xyz(1));
    fused_point_z_.push_back(xyz(2));
    fused_point_nx_.push_back(normal(0));
    fused_point_ny_.push_back(normal(1));
    fused_point_nz_.push_back(normal(2));
    fused_point_r_.push_back(color(0));
    fused_point_g_.push_back(color(1));
    fused_point_b_.push_back(color(2));

    // Remember the first pixel as the reference.
    if (traversal_depth == 0) {
      fused_ref_point = Eigen::Vector4f(xyz(0), xyz(1), xyz(2), 1.0f);
      fused_ref_normal = normal;
    }

    if (fused_point_x_.size() >= static_cast<size_t>(options_.max_num_pixels)) {
      //std::cout << "skip max fused " << std::endl;
      break;
    }

    FusionData next_data;
    next_data.traversal_depth = traversal_depth + 1;

    if (next_data.traversal_depth >= options_.max_traversal_depth) {
      //std::cout << "traversal_depth" << std::endl;
      continue;
    }

    for (const auto next_image_idx : overlapping_images_.at(image_idx)) {
      //std::cout << next_image_idx << std::endl;
      if (!used_images_.at(next_image_idx) ||
          fused_images_.at(next_image_idx)) {
            //std::cout << "skip used/fused " << std::endl;
        continue;
      }

      next_data.image_idx = next_image_idx;

      const cv::Vec3f cv_next_proj = Project(xyz_cv, Ks_[next_image_idx], Rs_[next_image_idx], ts_[next_image_idx]);
      const Eigen::Vector3f next_proj(cv_next_proj(0), cv_next_proj(1), cv_next_proj(2));

      next_data.col = static_cast<int>(std::round(next_proj(0) / next_proj(2)));
      next_data.row = static_cast<int>(std::round(next_proj(1) / next_proj(2)));

      const auto& depth_map_size = depth_map_sizes_.at(next_image_idx);
      if (next_data.col < 0 || next_data.row < 0 ||
          next_data.col >= depth_map_size.first ||
          next_data.row >= depth_map_size.second) {
          //std::cout << "skip outside " << std::endl;
        continue;
      }

      fusion_queue_.push_back(next_data);
    }
  }

  fusion_queue_.clear();

  const size_t num_pixels = fused_point_x_.size();
  if (num_pixels >= static_cast<size_t>(options_.min_num_pixels)) {
    Eigen::Vector3f fused_normal;
    fused_normal.x() = Median(&fused_point_nx_);
    fused_normal.y() = Median(&fused_point_ny_);
    fused_normal.z() = Median(&fused_point_nz_);
    const float fused_normal_norm = fused_normal.norm();
    if (fused_normal_norm < std::numeric_limits<float>::epsilon()) {
      //std::cout << "zero normal " << std::endl;
      return;
    }

    // fused_point.x = internal::Median(&fused_point_x_);
    // fused_point.y = internal::Median(&fused_point_y_);
    // fused_point.z = internal::Median(&fused_point_z_);
    merged_points->push_back(Median(&fused_point_x_));
    merged_points->push_back(Median(&fused_point_y_));
    merged_points->push_back(Median(&fused_point_z_));

    // fused_point.nx = fused_normal.x() / fused_normal_norm;
    // fused_point.ny = fused_normal.y() / fused_normal_norm;
    // fused_point.nz = fused_normal.z() / fused_normal_norm;
    fused_normal /= fused_normal_norm;
    merged_normals->push_back(fused_normal[0]);
    merged_normals->push_back(fused_normal[1]);
    merged_normals->push_back(fused_normal[2]);

    // fused_point.r = TruncateCast<float, uint8_t>(
    //     std::round(internal::Median(&fused_point_r_)));
    // fused_point.g = TruncateCast<float, uint8_t>(
    //     std::round(internal::Median(&fused_point_g_)));
    // fused_point.b = TruncateCast<float, uint8_t>(
    //     std::round(internal::Median(&fused_point_b_)));
    merged_colors->push_back(std::round(Median(&fused_point_r_)));
    merged_colors->push_back(std::round(Median(&fused_point_g_)));
    merged_colors->push_back(std::round(Median(&fused_point_b_)));

    // TODO : support semantic info
    merged_labels->push_back(0);
    merged_detections->push_back(0);
  }
}