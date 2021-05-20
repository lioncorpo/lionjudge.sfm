#pragma once
#include <Eigen/Core>
#include <vector>
// #include <opencv2/core.hpp>
#include <slam/guided_matching.h>
#include <unordered_set>
namespace map
{
  class Shot;
  class Landmark;
}

namespace slam
{
class SlamUtilities
{
public:
  static bool check_epipolar_constraint(const Eigen::Vector3f& bearing_1, const Eigen::Vector3f& bearing_2,
                                        const Eigen::Matrix3f& E_12, const float bearing_1_scale_factor);

  static Eigen::Matrix3d to_skew_symmetric_mat(const Eigen::Vector3d& vec);

  
  // static Eigen::Matrix3f create_E_21(const Eigen::Matrix3f& rot_1w, const Eigen::Vector3f& trans_1w,
  //                                    const Eigen::Matrix3f& rot_2w, const Eigen::Vector3f& trans_2w);

  static Eigen::Matrix3d create_E_21(const Eigen::Matrix3d &rot_1w, const Eigen::Vector3d &trans_1w,
                                     const Eigen::Matrix3d &rot_2w, const Eigen::Vector3d &trans_2w);


  // static Eigen::MatrixXf ConvertOpenCVKptsToEigen(const std::vector<cv::KeyPoint>& keypts);
  static Eigen::MatrixXf ConvertOpenCVKptsToEigen(const AlignedVector<map::Observation>& keypts);
  

  static std::vector<map::Landmark*> update_local_landmarks(const std::vector<map::Shot*>& local_keyframes); //, const size_t curr_frm_id);

  static std::vector<map::Shot*> update_local_keyframes(const map::Shot& curr_shot);

  static size_t MatchShotToLocalMap(map::Shot &curr_shot, const slam::GuidedMatcher& matcher);

  static void SetDescriptorFromObservations(map::Landmark& landmark);
  static void SetDescriptorFromObservationsEig(map::Landmark& landmark);
  static void SetNormalAndDepthFromObservations(map::Landmark& landmark, const std::vector<float>& scale_factors);

  static std::pair<double, double> ComputeMinMaxDepthInShot(const map::Shot& shot);

  static void FuseDuplicatedLandmarks(map::Shot& shot, const std::vector<map::Shot*>& fuse_shots, const slam::GuidedMatcher& matcher, const float margin,
                                      map::Map& slam_map);

  // static std::set<map::Shot*, map::KeyCompare> 
  static std::vector<map::Shot*> GetSecondOrderCovisibilityForShot(const map::Shot& shot, const size_t first_order_thr, const size_t second_order_thr);                                      

  static std::unordered_map<map::ShotId, map::Shot*> ComputeLocalKeyframes(map::Shot& shot);
};
} // namespace map