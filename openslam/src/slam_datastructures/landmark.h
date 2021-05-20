#pragma once
#include <map>
#include <opencv2/core.hpp>
#include <Eigen/Eigen>
#include <iostream>
#include "slam_utilities.h"
#include "keyframe.h"
namespace cslam
{
// class KeyFrame;
class Frame;
class SlamReconstruction;
class Landmark
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Landmark(const size_t lm_id, KeyFrame* ref_kf, const Eigen::Vector3f& pos_w);

    void increase_num_observable(unsigned int num_observable = 1){ num_observable_ += num_observable; }
    void increase_num_observed(unsigned int num_observed = 1) { num_observed_ += num_observed; }
    float get_observed_ratio() const { return static_cast<float>(num_observed_)/num_observable_; }


    void erase_observation(KeyFrame* kf, SlamReconstruction* reconstruction);

    //! whether this landmark will be erased shortly or not
    bool will_be_erased() { return will_be_erased_; };
    cv::Mat get_descriptor() const { 
        // std::cout << "Returning desc " << descriptor_ << " lm_id: " << lm_id_ << " ptr: " << this << std::endl;
        return descriptor_.clone(); 
    }
    bool has_observation() const { return num_observations_ > 0; }
    size_t num_observations() const { return num_observations_; }
    void add_observation(KeyFrame*, const size_t idx);
    void compute_descriptor();
    void update_normal_and_depth(); //const std::vector<float>& scale_factors);


    Eigen::Vector3f get_pos_in_world() const { return pos_w_; }
    void set_pos_in_world(const Eigen::Vector3f& pos_w) { pos_w_ = pos_w; }
    float get_min_valid_distance() const { return 0.7 * min_valid_dist_; }
    float get_max_valid_distance() const { return 1.3 * max_valid_dist_; }
    //! predict scale level assuming this landmark is observed in the specified frame
    size_t predict_scale_level(const float cam_to_lm_dist, const Frame& frm) const;
    //! predict scale level assuming this landmark is observed in the specified keyframe
    size_t predict_scale_level(const float cam_to_lm_dist, const KeyFrame& keyfrm) const;
    size_t predict_scale_level(const float cam_to_lm_dist, const float log_scale_factor, const size_t num_scale_levels) const;
    //! replace this with specified landmark
    void replace(Landmark* lm);
    //! check the distance between landmark and camera is in ORB scale variance
    inline bool is_inside_in_orb_scale(const float cam_to_lm_dist) const {
        return (get_min_valid_distance() <= cam_to_lm_dist && cam_to_lm_dist <= get_max_valid_distance());
    }
    Eigen::Vector3f get_obs_mean_normal() const { return mean_normal_; }
    void prepare_for_erasing();
    //! whether this landmark is observed in the specified keyframe
    bool is_observed_in_keyframe(KeyFrame* keyfrm) const
    {
        return static_cast<bool>(observations_.count(keyfrm));
    }

    const auto& get_observations() { return observations_; }
    // const auto& get_observations() const { return observations_; }
    Landmark* get_replaced() const { return replaced_; }
public:
    const size_t lm_id_;
    //! reference keyframe
    KeyFrame* ref_keyfrm_;
    size_t ref_kf_id_;
    // Tracking information
    bool is_observable_in_tracking_ = false; // true if can be reprojected to current frame
    size_t scale_level_in_tracking_;
        
    size_t identifier_in_local_map_update_ = 0;
    size_t identifier_in_local_lm_search_ = 0;

    Eigen::Vector2f reproj_in_tracking_ = Eigen::Vector2f::Zero(); // reprojected pixel position

private:
    
    size_t num_observations_ = 0;

    // track counter
    size_t num_observable_ = 1;
    size_t num_observed_ = 1;

    // ORB scale variances
    //! max valid distance between landmark and camera
    float min_valid_dist_ = 0;
    //! min valid distance between landmark and camera
    float max_valid_dist_ = 0;

    //! この3次元点を観測しているkeyframeについて，keyframe->lmのベクトルの平均値(規格化されてる)
    Eigen::Vector3f mean_normal_ = Eigen::Vector3f::Zero();

    //! representative descriptor
    cv::Mat descriptor_;


    
    Eigen::Vector3f pos_w_;

    //! observations (keyframe and keypoint index)
    std::map<KeyFrame*, size_t, KeyFrameCompare> observations_;
    // std::map<KeyFrame*, size_t, decltype(SlamUtilities::compare)> observations_(SlamUtilities::compare);


    //! this landmark will be erased shortly or not
    bool will_be_erased_ = false; //Note: probably always false since we don't run in parallel
    Landmark* replaced_ = nullptr;
};
} // namespace cslam
