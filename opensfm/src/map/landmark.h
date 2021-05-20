#pragma once
#include <Eigen/Eigen>
#include <map>
#include <unordered_map>
#include <memory>
#include <map/defines.h>
#include <iostream>
// #include <map/shot.h>
namespace map
{
class Shot;

class SLAMLandmarkData{

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
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
  static LandmarkUniqueId landmark_unique_id_;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // Landmark(const LandmarkId lm_id, const Eigen::Vector3d& global_pos, const std::string& name = "");
  Landmark(const LandmarkId lm_id, const Eigen::Vector3d& global_pos);
  Eigen::Vector3d GetGlobalPos() const { return global_pos_; }
  void SetGlobalPos(const Eigen::Vector3d& global_pos) { global_pos_ = global_pos; }

  bool IsObservedInShot(Shot* shot) const { return observations_.count(shot) > 0; }
  
  void AddObservation(Shot* shot, const FeatureId feat_id) { observations_.emplace(shot, feat_id); }
  void RemoveObservation(Shot* shot) { observations_.erase(shot); }
  bool HasObservations() const { return !observations_.empty(); }
  auto NumberOfObservations() const { return observations_.size(); }
  Eigen::Vector3f GetObservationInShot(Shot* shot) const;
  const auto& GetObservations() const
  {
    return observations_; 
  }
  void ClearObservations() { observations_.clear(); }
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
  void SetReprojectionErrors(const std::unordered_map<std::string, Eigen::VectorXd> reproj_errors);
  auto GetReprojectionErrors() const { return reproj_errors_; }
  void RemoveReprojectionError(const std::string& shot_id)
  {
    reproj_errors_.erase(shot_id);
  }

public:
  //We could set the const values to public, to avoid writing a getter.
  const LandmarkId id_;
  const LandmarkUniqueId unique_id_;
  // const std::string name_;
  SLAMLandmarkData slam_data_;
private:
  Eigen::Vector3d global_pos_; // point in global
  std::map<Shot*, FeatureId, KeyCompare> observations_;
  Shot* ref_shot_; //shot in which the landmark was first seen
  Eigen::Vector3i color_;
  std::unordered_map<std::string, Eigen::VectorXd> reproj_errors_;

};
}