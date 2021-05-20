#pragma once
#include <Eigen/Eigen>
#include <map>
#include <memory>
#include <map/defines.h>
#include <iostream>
// #include <opencv2/core.hpp>
namespace map
{
class Shot;

class SLAMLandmarkData{
public:
  // cv::Mat descriptor_;
  DescriptorType descriptor_;
  size_t num_observations_ = 0;

  Eigen::Vector3d mean_normal_ = Eigen::Vector3d::Zero();
  float GetMinValidDistance() const { return 0.7 * min_valid_dist_; }
  float GetMaxValidDistance() const { return 1.3 * max_valid_dist_; }
  void IncreaseNumObservable(unsigned int num_observable = 1){ num_observable_ += num_observable; }
  void IncreaseNumObserved(unsigned int num_observed = 1) { num_observed_ += num_observed; }
  float GetObservedRatio() const { return static_cast<float>(num_observed_)/num_observable_; }
  // ORB scale variances
  //! max valid distance between landmark and camera
  float min_valid_dist_ = 0;
  //! min valid distance between landmark and camera
  float max_valid_dist_ = 0; 
  auto GetNumObserved() const { return num_observed_; }
  auto GetNumObservable() const { return num_observable_; }
private:
  // track counter
  size_t num_observable_ = 1;
  size_t num_observed_ = 1;
};

class Landmark {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Landmark(const LandmarkId lm_id, const Eigen::Vector3d& global_pos, const std::string& name = "");
  Eigen::Vector3d GetGlobalPos() const { return global_pos_; }
  void SetGlobalPos(const Eigen::Vector3d& global_pos) { global_pos_ = global_pos; }

  bool IsObservedInShot(Shot* shot) const { return observations_.count(shot) > 0; }
  
  void AddObservation(Shot* shot, const FeatureId feat_id) { observations_.emplace(shot, feat_id); }
  void RemoveObservation(Shot* shot) { observations_.erase(shot); }
  bool HasObservations() const { return !observations_.empty(); }
  auto NumberOfObservations() const { return observations_.size(); }
  const auto& GetObservations() const
  {
    return observations_; 
  }
  void SetRefShot(Shot* ref_shot) {ref_shot_ = ref_shot;}
  Shot* GetRefShot() { return ref_shot_; }
  double ComputeDistanceFromRefFrame() const;
  Eigen::Vector3i GetColor() const { return color_; }
  void SetColor(const Eigen::Vector3i& color) { color_ = color; }
  //Comparisons
  bool operator==(const Landmark& lm) const { return id_ == lm.id_; }
  bool operator!=(const Landmark& lm) const { return !(*this == lm); }
  bool operator<(const Landmark& lm) const { return id_ < lm.id_; }
  bool operator<=(const Landmark& lm) const { return id_ <= lm.id_; }
  bool operator>(const Landmark& lm) const { return id_ > lm.id_; }
  bool operator>=(const Landmark& lm) const { return id_ >= lm.id_; }
public:
  //We could set the const values to public, to avoid writing a getter.
  const LandmarkId id_;
  const std::string name_;
  SLAMLandmarkData slam_data_;
private:
  Eigen::Vector3d global_pos_; // point in global
  std::map<Shot*, FeatureId, KeyCompare> observations_;
  Shot* ref_shot_; //shot in which the landmark was first seen
  Eigen::Vector3i color_;
};
}