#include <algorithm>
#include "point.h"

void
Pose::SetPose(const Pose& pose)
{
  worldToCam_ = pose.WorldToCamera();
  camToWorld_ = pose.CameraToWorld();
}

Point::Point(const PointId point_id, const Eigen::Vector3d& global_pos, const std::string& name):
  id_(point_id), global_pos_(global_pos), point_name_(name)
{

}

bool 
Point::IsObservedInShot(Shot* shot) const
{
  return observations_.count(shot);
}
void 
Point::AddObservation(Shot* shot, const FeatureId feat_id)
{
  observations_.emplace(shot, feat_id);
}
void
Point::HasObservations() const
{
  return !observations_.empty();
}
void
Point::RemoveObservation(Shot* shot)
{
  observations_.erase(shot);
}

Shot::Shot(const ShotId shot_id, const camera* camera, const Pose& pose, const std::string& name):
            id_(shot_id), image_name(name), camera_(camera), pose_(pose)
{
  
}

void
Shot::RemovePointObservation(const FeatureId id)
{
  points_.at(id) = nullptr;
}

size_t
Shot::ComputeNumValidPoints() const
{
  return points_.size() - std::count(points_.cbegin(), points_.cend(), nullptr);
}

Shot*
ReconstructionManager::CreateShot(const ShotId shot_id, const CameraId camera_id, const Pose& pose, const std::string& name)
{
  const auto* shot_cam = cameras_.at(camera_id);
  auto it = shots_.emplace(shot_id, std::make_unique<Shot>(shot_id, shot_cam, pose, name));
  
  // Insert failed
  if (!it.second)
  {
    return nullptr;

  }

  if (!name.empty())
  {  
    shot_names_.emplace(name, shot_id);
  }
  return it.second->get();
}

void
ReconstructionManager::UpdateShotPose(const ShotId shot_id, const Pose& pose)
{
  shots_.at(shot_id).SetPose(pose);
  return true;
}

Point*
ReconstructionManager::CreatePoint(const PointId point_id, const Eigen::Vector3d& global_pos, const std::string& name = "")
{

  auto it = points.emplace(point_id, std::make_unique<Point>(point_id, global_pos, name));
  
  // Insert failed
  if (!it.second)
  {
    return nullptr;

  }

  if (!name.empty())
  {  
    point_names_.emplace(name, point_id);
  }
  return it.second->get();
}

void
ReconstructionManager::UpdatePoint(const PointId point_id, const Eigen::Vector3d& global_pos)
{
  points.at(point_id).setPointInGlobal(pose);
}

void 
ReconstructionManager::AddObservation(const Shot* shot, const Point* point, const FeatureId feat_id)
{
  shot->AddPointObservation(point, feat_id);
  point->AddObservation(shot, feat_id);
}

void
ReconstructionManager::RemoveObservation(const Shot* shot, const Point* point, const FeatureId feat_id)
{
  shot->RemovePointObservation(feat_id);
  point->RemoveObservation(shot);
}
