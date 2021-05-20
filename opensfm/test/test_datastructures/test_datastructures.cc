#include <map/map.h>
#include <map/landmark.h>
#include <map/camera.h>
#include <map/shot.h>

#include <iostream>
#include <vector>

void
test_shot_cameras(const size_t n_cameras, map::Manager& map_manager, const map::Camera* const cam_model)
{
  std::vector<map::ShotCamera*> cameras;
  if (n_cameras == 0)
  {
    return;
  }
  
  cameras.reserve(n_cameras);
  for (size_t i = 0; i < n_cameras; ++i)
  {
    cameras.push_back(map_manager.CreateShotCamera(i, *cam_model, "cam"+std::to_string(i)));
  }
  for (const auto& shot_cam : cameras)
  {
    std::cout << "shot_cam: " << shot_cam->id_ << " name: " << shot_cam->camera_name_ << std::endl;
  }

  std::cout << "-------------------------" << std::endl;
  std::cout << "Number of Cams: " << map_manager.NumberOfCameras() 
            << " all cams: " << map_manager.GetAllCameras().size() << std::endl;
  const auto id_0 = cameras[0]->id_;
  map_manager.RemoveShotCamera(id_0);
  map_manager.RemoveShotCamera(id_0);
  map_manager.RemoveShotCamera(id_0);
  map_manager.RemoveShotCamera(cameras[5]->id_);
  std::cout << "Number of Cams after del: " << map_manager.NumberOfCameras() << std::endl;
}

void
test_large_problem(map::Manager& map_manager, const map::Camera& cam)
{
  std::vector<map::ShotCamera*> cameras;
  constexpr size_t n_cameras{2};
  for (size_t i = 0; i < n_cameras; ++i)
  {
    cameras.push_back(map_manager.CreateShotCamera(i, cam, "cam"+std::to_string(i)));
  }
  constexpr size_t n_observations{100};
  std::vector<map::Shot*> shots;
  //Create ten shots with two different cameras
  for (size_t i = 0; i < 10; ++i)
  {
    if (i < 5)
    {
      shots.push_back(map_manager.CreateShot(i, *cameras[0]));
    }
    else
    {
      shots.push_back(map_manager.CreateShot(i, *cameras[1]));
    }
    shots.back()->InitKeyptsAndDescriptors(n_observations);
  }

  constexpr auto n_landmarks{200};
  std::vector<map::Landmark*> landmarks;
  for (size_t lm_id = 0; lm_id < n_landmarks; ++lm_id)
  {
    landmarks.push_back(map_manager.CreateLandmark(lm_id, Eigen::Vector3d(0,0,0),"lm"));
  }
  
  // assign 100 to each shot (observations)
  for (auto& shot : shots)
  {
    Eigen::VectorXi lm_obs = (n_observations*Eigen::VectorXf::Random(n_observations).array()+n_observations).cast<int>();
    Eigen::VectorXi feat_obs = (n_observations/2*Eigen::VectorXf::Random(n_observations).array()+n_observations/2).cast<int>();
    for (size_t idx = 0; idx < n_observations; ++idx)
    {
      // std::cout << "shot: " << shot->id_ << " lm_obs["<<idx<<"]" << ", feat_obs["<<idx<<"]" << std::endl;
      // std::cout << "lm_obs: " << lm_obs << std::endl;
      // std::cout << "feat_obs" << feat_obs << std::endl;
      // std::cout << "landmarks[lm_obs[idx]]: " << landmarks[lm_obs[idx]] << std::endl;
      // std::cout << "feat_obs[idx]: " << feat_obs[idx] << std::endl;
      map_manager.AddObservation(shot, landmarks[lm_obs[idx]], feat_obs[idx]);
    }
    std::cout << "shot->ComputeNumValidLandmarks" << shot->ComputeNumValidLandmarks() << std::endl;
  }

  auto& shot_landmarks1 = shots[0]->GetLandmarks();
  // auto& shot_landmarks2 = shots[1]->GetLandmarks();
  if (!shot_landmarks1.empty())
  {
    const auto& obs_shot1 = shot_landmarks1[0]->GetObservations();
    std::cout << "Observations before delete!" << std::endl;
    for (const auto& p : obs_shot1)
    {
      const auto& shot = p.first;
      std::cout << shot->id_ << " name: " << shot->name_ << " #lms: " << shot->ComputeNumValidLandmarks() << std::endl;
    }
    const auto copy_map(obs_shot1);
    map_manager.RemoveLandmark(shot_landmarks1[0]);
    
    std::cout << "Observations after delete!" << std::endl;
    for (const auto& p : copy_map)
    {
      const auto& shot = p.first;
      std::cout << shot->id_ << " name: " << shot->name_ << " #lms: " << shot->ComputeNumValidLandmarks() << std::endl;
    }
  }
}


int main()
{
  map::Manager map_manager;
  std::cout << "map_manager:" << map_manager.NumberOfCameras()
            << ", " << map_manager.NumberOfLandmarks() << std::endl;
  map::Camera cam;
  test_shot_cameras(10, map_manager, &cam);

  test_large_problem(map_manager, cam);
  return 0;
}