#include <map/map.h>
#include <map/landmark.h>
#include <map/camera.h>
#include <map/shot.h>
#include <map/pose.h>
#include <unordered_set>

namespace map
{

void 
Map::AddObservation(Shot *const shot,  Landmark *const lm, const FeatureId feat_id) const
{
  shot->AddLandmarkObservation(lm, feat_id);
  lm->AddObservation(shot, feat_id);
}

void
Map::RemoveObservation(Shot *const shot,  Landmark *const lm, const FeatureId feat_id) const
{
  shot->RemoveLandmarkObservation(feat_id);
  lm->RemoveObservation(shot);
}


/**
 * Creates a shot and returns a pointer to it
 * 
 * @param shot_id       unique id of the shot
 * @param camera_id     unique id of EXISTING camera
 * @param global_pos    pose in the 3D world
 * @param name          name of the shot
 * 
 * @returns             returns pointer to created or existing shot
 */
Shot*
Map::CreateShot(const ShotId shot_id, const ShotCamera& shot_cam, const std::string& name, const Pose& pose)
{
  auto it = shots_.emplace(shot_id, std::make_unique<Shot>(shot_id, shot_cam, pose, name));

  if (!name.empty())
  {  
    shot_names_.emplace(name, shot_id);
  }
  if (it.second) //only if insert really happened
  {
    // prevent problems when, e.g. [1,2,3] are present 
    // and id 1000 comes in!
    unique_shot_id_ = std::max(shot_id+1, unique_shot_id_);
  } 
  return it.first->second.get();
}

Shot*
Map::CreateShot(const ShotId shot_id, const CameraId camera_id, const std::string& name, const Pose& pose)
{
  return CreateShot(shot_id, *cameras_.at(camera_id), name, pose);
}

void
Map::UpdateShotPose(const ShotId shot_id, const Pose& pose)
{
  shots_.at(shot_id)->SetPose(pose);
}

void 
Map::RemoveShot(const ShotId shot_id)
{
    //1) Find the point
  const auto& shot_it = shots_.find(shot_id);
  if (shot_it != shots_.end())
  {
    const auto& shot = shot_it->second;
    //2) Remove it from all the points
    for (const auto& lm : shot->GetLandmarks())
    {
      if (lm != nullptr)
      {
        lm->RemoveObservation(shot.get());
      }
    }

    //3) Remove from shot_names
    shot_names_.erase(shot->name_);

    //4) Remove from shots
    shots_.erase(shot_it);
  }
}


/**
 * Creates a landmark and returns a pointer to it
 * 
 * @param lm_Id       unique id of the landmark
 * @param global_pos  3D position of the landmark
 * @param name        name of the landmark
 * 
 * @returns           pointer to the created or already existing lm
 */
Landmark*
Map::CreateLandmark(const LandmarkId lm_id, const Eigen::Vector3d& global_pos, const std::string& name)
{
  auto it = landmarks_.emplace(lm_id, std::make_unique<Landmark>(lm_id, global_pos, name));
  if (!name.empty())
  {  
    landmark_names_.emplace(name, lm_id);
  }
  if (it.second) //only if insert really happened
  {
    // prevent problems when, e.g. [1,2,3] are present 
    // and id 1000 comes in!
    unique_landmark_id_ = std::max(lm_id+1, unique_landmark_id_);
  }
  return it.first->second.get(); //the raw pointer
}

void
Map::UpdateLandmark(const LandmarkId lm_id, const Eigen::Vector3d& global_pos)
{
  landmarks_.at(lm_id)->SetGlobalPos(global_pos);
}
void 
Map::RemoveLandmark(const Landmark* const lm)
{
  if (lm != nullptr)
  {
    //2) Remove all its observation
    const auto& observations = lm->GetObservations();
    for (const auto& obs : observations)
    {
      Shot* shot = obs.first;
      const auto feat_id = obs.second;
      shot->RemoveLandmarkObservation(feat_id);
    }

        //3) Remove from landmark_names_
    landmark_names_.erase(lm->name_);

    //4) Remove from landmarks
    landmarks_.erase(lm->id_);
  }
}
void 
Map::RemoveLandmark(const LandmarkId lm_id)
{
  //1) Find the landmark
  const auto& lm_it = landmarks_.find(lm_id);
  if (lm_it != landmarks_.end())
  {
    const auto& landmark = lm_it->second;
    //2) Remove all its observation
    const auto& observations = landmark->GetObservations();
    for (const auto& obs : observations)
    {
      Shot* shot = obs.first;
      const auto feat_id = obs.second;
      shot->RemoveLandmarkObservation(feat_id);
    }

    //3) Remove from landmark_names_
    landmark_names_.erase(landmark->name_);

    //4) Remove from landmarks
    landmarks_.erase(lm_it);
  }
}
/**
 * Replaces landmark old_lm by new_lm
 * 
 * 
 * 
 * 
 */

void 
Map::ReplaceLandmark(Landmark* old_lm, Landmark* new_lm)
{
  if (old_lm == nullptr || new_lm == nullptr || old_lm->id_ == new_lm->id_)
  {
    return;
  }
  // go through the observations of old_lm
  for (const auto& observation : old_lm->GetObservations())
  {
    Shot* obs_shot = observation.first;
    FeatureId obs_feat_id = observation.second;
    Landmark* test_lm2 = obs_shot->GetLandmark(obs_feat_id);
    if (test_lm2 != old_lm)
    {
      std::cout << "id failure" << obs_feat_id << std::endl;
      exit(0);
    }
    {
      //check if the old_lm is still somewhere
      const auto& obs_lms = obs_shot->GetLandmarks();
      std::unordered_map<Landmark*,FeatureId> test_set;
      for (size_t i = 0; i < obs_shot->NumberOfKeyPoints(); ++i)
      {
        
        auto* lm = obs_lms[i];
        if (lm != nullptr)
        {
          auto it = test_set.find(lm);
          if (it == test_set.end())
          {
            //insert
            test_set[lm] = i;
          }
          else
          {
            std::cout << "Double landmark " << lm->id_ << " match in shot " << obs_shot->name_ << " at " << i << " <-> " << it->second<< std::endl;
            std::cout<< obs_shot->GetLandmark(i) << ", " << obs_shot->GetLandmark(it->second) << std::endl;
          }
        }
        // if (lm == old_lm)
        // {
        //   std::cout << "Not replaced correctly!!! at " << i << std::endl;
        //   exit(0);
        // }
      }
    }

    // if the new one is seen in obs_shot, there was a mismatch
    // Thus, erase the observation of the old_lm
    if (new_lm->IsObservedInShot(obs_shot))
    {
      obs_shot->RemoveLandmarkObservation(obs_feat_id);
      std::cout << "removing old lm " << old_lm->id_ << " at " << obs_feat_id 
                << " at shot: " << obs_shot->name_ << "/" << obs_shot->id_<< std::endl;
    }
    else
    {
      // replace, should be the same as AddObservation
      obs_shot->AddLandmarkObservation(new_lm, obs_feat_id);
      new_lm->AddObservation(obs_shot, obs_feat_id);
      std::cout << "replacing old lm " <<  old_lm->id_ << "at "<< obs_feat_id 
                << " at shot: " << obs_shot->name_ << "/" << obs_shot->id_<< std::endl;
    }
    //check if the old_lm is still somewhere
    const auto& obs_lms = obs_shot->GetLandmarks(); 
    for (size_t i = 0; i < obs_shot->NumberOfKeyPoints(); ++i)
    {
      const auto* lm = obs_lms[i];
      if (lm == old_lm)
      {
        std::cout << "Not replaced correctly!!! at " << i << std::endl;
        exit(0);
      }
    }
    // just for testing
    auto* test_lm = obs_shot->GetLandmark(obs_feat_id);
    if (test_lm == old_lm)
    {
      std::cout << "Error during replace!" << std::endl;
      exit(0);
    }
  }
  // TODO: This basically takes all the observations from the old lm
  // Might not be completely correct
  new_lm->slam_data_.IncreaseNumObserved(old_lm->slam_data_.GetNumObserved());
  new_lm->slam_data_.IncreaseNumObservable(old_lm->slam_data_.GetNumObservable());
  std::cout << "Erasing: " << old_lm->id_ << " and replacing with" << new_lm->id_ << std::endl;
  //3) Remove from landmark_names_
  landmark_names_.erase(old_lm->name_);
  //4) Remove from landmarks
  landmarks_.erase(old_lm->id_);

}

/**
 * Creates a shot camera and returns a pointer to it
 * 
 * @param cam_id       unique id of the shot camera
 * @param Camera  3D position of the landmark
 * @param name        name of the landmark
 * 
 * @returns           pointer to the created or already existing lm
 */
ShotCamera* 
Map::CreateShotCamera(const CameraId cam_id, const Camera& camera, const std::string& name)
{
  auto it = cameras_.emplace(cam_id, std::make_unique<ShotCamera>(camera, cam_id, name));
  
  if (!name.empty())
  {  
    camera_names_.emplace(name, cam_id);
  }
  return it.first->second.get();
}


void
Map::RemoveShotCamera(const CameraId cam_id)
{
  const auto& cam_it = cameras_.find(cam_id);
  if (cam_it != cameras_.end())
  {
    cameras_.erase(cam_it);
  }
}

};
