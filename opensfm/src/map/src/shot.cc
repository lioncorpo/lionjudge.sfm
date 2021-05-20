#include <map/shot.h>
#include <map/camera.h>
#include <map/landmark.h>
#include <algorithm>
#include <numeric>
namespace map
{
Shot::Shot(const ShotId shot_id, const ShotCamera& shot_camera, const Pose& pose, const std::string& name):
            id_(shot_id), name_(name), shot_camera_(shot_camera), slam_data_(this), pose_(pose)
{
  
}

size_t
Shot::ComputeNumValidLandmarks(const int min_obs_thr) const
{
  return std::accumulate(landmarks_.cbegin(), landmarks_.cend(), 0,
                    [min_obs_thr](const auto prior, const auto* lm)
                    {
                        if (lm != nullptr && min_obs_thr <= lm->NumberOfObservations())
                          return prior + 1;
                        return prior;
                    });
  // return landmarks_.size() - std::count(landmarks_.cbegin(), landmarks_.cend(), nullptr);
}

float
Shot::ComputeMedianDepthOfLandmarks(const bool take_abs) const
{
  if (landmarks_.empty())
    return 1.0f;
  std::vector<float> depths;
  depths.reserve(landmarks_.size());
  const Eigen::Matrix4d T_cw = pose_.WorldToCamera();
  const Eigen::Vector3d rot_cw_z_row = T_cw.block<1, 3>(2, 0);

  // T_cw
  const double trans_cw_z = T_cw(2, 3);
  for (const auto& lm : landmarks_)
  {
      if (lm != nullptr)
      {
        const double pos_c_z = rot_cw_z_row.dot(lm->GetGlobalPos())+trans_cw_z;
        depths.push_back(float(take_abs ? std::abs(pos_c_z) : pos_c_z));
      }
  }
  std::sort(depths.begin(), depths.end());
  return depths.at((depths.size() - 1) / 2);
}

void
Shot::InitKeyptsAndDescriptors(const size_t n_keypts)
{
  if (n_keypts > 0)
  {
    num_keypts_ = n_keypts;
    landmarks_.resize(num_keypts_, nullptr);
    keypoints_.resize(num_keypts_);
    // descriptors_ = cv::Mat(n_keypts, 32, CV_8UC1, cv::Scalar(0));
    descriptors_ = DescriptorMatrix(n_keypts, 32);
  }
}

void
// Shot::InitAndTakeDatastructures(AlignedVector<Observation> keypts, cv::Mat descriptors)
Shot::InitAndTakeDatastructures(AlignedVector<Observation> keypts, DescriptorMatrix descriptors)
{
  assert(keypts.size() == descriptors.rows);

  std::swap(keypts, keypoints_);
  std::swap(descriptors, descriptors_);
  num_keypts_ = keypoints_.size();
  landmarks_.resize(num_keypts_, nullptr);
}

void
Shot::UndistortedKeyptsToBearings()
{
  if (!slam_data_.undist_keypts_.empty())
  {
    shot_camera_.camera_model_.UndistortedKeyptsToBearings(slam_data_.undist_keypts_, slam_data_.bearings_);
  }
}

void
Shot::UndistortKeypts()
{
  if (!keypoints_.empty())
  {
    shot_camera_.camera_model_.UndistortKeypts(keypoints_, slam_data_.undist_keypts_);
  }
}

void
Shot::ScaleLandmarks(const float scale)
{
  for (auto lm : landmarks_) 
  {
    if (lm != nullptr)
    {
      lm->SetGlobalPos(lm->GetGlobalPos()*scale);
    }
  }
}

void
Shot::ScalePose(const float scale)
{
    Eigen::Matrix4d cam_pose_cw = pose_.WorldToCamera();
    cam_pose_cw.block<3, 1>(0, 3) *= scale;
    pose_.SetFromWorldToCamera(cam_pose_cw);
}

} //namespace map

