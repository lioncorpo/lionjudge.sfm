#pragma once
#include <map/defines.h>
#include <Eigen/Core>
#include <unordered_map>
#include <map>
#include <memory>

#include <map/defines.h>
#include <map/pose.h>
namespace map
{
class Shot;
class Landmark;
class ShotCamera;
class Camera;

class Map 
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Should belong to the Map
  ShotId GetShotIdFromName(const std::string& name) const { return shot_names_.at(name); }
  LandmarkId GetPointIdFromName(const std::string& name) const { return landmark_names_.at(name); };

  // Map information
  auto NumberOfShots() const { return shots_.size(); }
  auto NumberOfLandmarks() const { return landmarks_.size(); }
  auto NumberOfCameras() const { return cameras_.size(); }

  // Create, Update and Remove
  // Camera
  ShotCamera* CreateShotCamera(const CameraId cam_id, const Camera& camera, const std::string& name = "");
  void UpdateShotCamera(const CameraId cam_id, const Camera& camera);
  void RemoveShotCamera(const CameraId cam_id);

  // Shots
  Shot* CreateShot(const ShotId shot_id, const CameraId camera_id, const std::string& name = "", const Pose& pose = Pose());
  Shot* CreateShot(const ShotId shot_id, const ShotCamera& shot_cam, const std::string& name = "", const Pose& pose = Pose());
  void UpdateShotPose(const ShotId shot_id, const Pose& pose);
  void RemoveShot(const ShotId shot_id);
  auto GetNextUniqueShotId() const { return unique_shot_id_; }
  Shot* GetShot(const ShotId shot_id) const { 
    const auto& it = shots_.find(shot_id);
    return (it != shots_.end() ? it->second.get() : nullptr);
  }
  
  // Landmark
  Landmark* CreateLandmark(const LandmarkId lm_id, const Eigen::Vector3d& global_pos, const std::string& name = "");
  void UpdateLandmark(const LandmarkId lm_id, const Eigen::Vector3d& global_pos);
  void RemoveLandmark(const Landmark* const lm);
  void RemoveLandmark(const LandmarkId lm_id);
  auto GetNextUniqueLandmarkId() const { return unique_landmark_id_; }
  void ReplaceLandmark(Landmark* old_lm, Landmark* new_lm);
  void AddObservation(Shot *const shot,  Landmark *const lm, const FeatureId feat_id) const;
  void RemoveObservation(Shot *const shot,  Landmark *const lm, const FeatureId feat_id) const;

  std::map<Landmark*, FeatureId> GetObservationsOfShot(const Shot* shot);
  std::map<Shot*, FeatureId> GetObservationsOfPoint(const Landmark* point);  

  const auto& GetAllShots() const { return shots_; }
  const auto& GetAllCameras() const { return cameras_; };
  const auto& GetAllLandmarks() const { return landmarks_; };

  const auto GetAllShotPointers() const
  {
    std::unordered_map<ShotId, Shot*> copy;
    std::transform(shots_.begin(), shots_.end(), std::inserter(copy, copy.end()), [](auto& elem) { return std::make_pair(elem.first, elem.second.get()); });
    return copy;
  }
  const auto GetAllCameraPointers() const
  {
    std::unordered_map<CameraId, ShotCamera*> copy;
    std::transform(cameras_.begin(), cameras_.end(), std::inserter(copy, copy.end()), [](auto& elem) { return std::make_pair(elem.first, elem.second.get()); });
    return copy;
  }
  const auto GetAllLandmarkPointers() const
  {
    std::unordered_map<LandmarkId, Landmark*> copy;
    std::transform(landmarks_.begin(), landmarks_.end(), std::inserter(copy, copy.end()), [](auto& elem) { return std::make_pair(elem.first, elem.second.get()); });
    return copy;
  }

private:
  std::unordered_map<CameraId, std::unique_ptr<ShotCamera>> cameras_;
  std::unordered_map<ShotId, std::unique_ptr<Shot>> shots_;
  std::unordered_map<LandmarkId, std::unique_ptr<Landmark>> landmarks_;
  std::unordered_map<std::string, ShotId> shot_names_;
  std::unordered_map< std::string, LandmarkId> landmark_names_;
  std::unordered_map<std::string, CameraId> camera_names_;

  size_t unique_shot_id_ = 0;
  size_t unique_landmark_id_ = 0;
};

} // namespace map