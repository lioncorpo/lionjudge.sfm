#pragma once
#include <vector>
#include <Eigen/Eigen>
#include <opencv2/core.hpp>
#include <pybind11/pybind11.h>
#include "types.h"
#include "third_party/openvslam/data/graph_node.h"
namespace py = pybind11;

namespace cslam
{
class Frame;
class Landmark;
class SlamReconstruction;


class KeyFrame
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    KeyFrame() = delete;
    KeyFrame(const size_t kf_id, const Frame& frame);

    std::vector<Landmark*> get_valid_lms();
    std::vector<size_t> get_valid_idx() const;

    
    Eigen::Vector3f get_obs_by_idx(const size_t idx) const
    {
        const auto kpt = keypts_[idx];
        return Eigen::Vector3f(kpt.pt.x, kpt.pt.y, kpt.size);
    }
    
    Eigen::Matrix4f get_Twc() const { return T_wc; }
    
    void set_Twc(const Eigen::Matrix4f& T_wc_)
    {
        T_wc = T_wc_;
        T_cw = T_wc.inverse();
        cam_center_ = T_wc.block<3,1>(0,3);
    }
    
    void set_Tcw(const Eigen::Matrix4f& T_cw_)
    {
        T_cw = T_cw_;
        T_wc = T_cw_.inverse();
        cam_center_ = T_wc.block<3,1>(0,3);
    }

    //The way it looks, this is the world pose
    Eigen::Matrix4f get_Tcw() const { return T_cw; }
    Eigen::Vector3f get_cam_center() const { return cam_center_; }
    Eigen::Matrix3f get_rotation() const { return T_cw.block<3,3>(0,0); }
    Eigen::Vector3f get_translation() const { return T_cw.block<3,1>(0,3); }

    py::object get_valid_keypts() const;
    
    void add_landmark(Landmark* lm, const size_t idx);
    size_t get_num_tracked_lms(const size_t min_num_obs_thr) const;
    // basically store_new_keyframe
    // std::vector<Landmark*> update_lms_after_kf_insert();
    void erase_landmark_with_index(const unsigned int idx) 
    {
        // std::lock_guard<std::mutex> lock(mtx_observations_);
        landmarks_.at(idx) = nullptr;
    }
    float compute_median_depth(const bool abs) const;
    py::object getKptsUndist() const;
    py::object getKptsPy() const;
    py::object getDescPy() const;


    /**
     * Erase a landmark observed by myself at keypoint idx
     */
    void erase_landmark_with_index(const size_t idx)
    {
        landmarks_.at(idx) = nullptr;
    }

    std::vector<size_t> compute_local_keyframes() const;

    /**
     * Whether this keyframe will be erased shortly or not
     */
    bool will_be_erased() { return will_be_erased_; }

    /**
     * Replace the landmark
     */
    void replace_landmark(Landmark* lm, const size_t idx)
    {
        landmarks_.at(idx) = lm;
    }

    void apply_landmark_replace();

    // void set_not_to_be_erased() { cannot_be_erased_ = True; }
        // operator overrides
    bool operator==(const KeyFrame& keyfrm) const { return kf_id_ == keyfrm.kf_id_; }
    bool operator!=(const KeyFrame& keyfrm) const { return !(*this == keyfrm); }
    bool operator<(const KeyFrame& keyfrm) const { return kf_id_ < keyfrm.kf_id_; }
    bool operator<=(const KeyFrame& keyfrm) const { return kf_id_ <= keyfrm.kf_id_; }
    bool operator>(const KeyFrame& keyfrm) const { return kf_id_ > keyfrm.kf_id_; }
    bool operator>=(const KeyFrame& keyfrm) const { return kf_id_ >= keyfrm.kf_id_; }
    // bool operator<(KeyFrame* const& kf1, KeyFrame* const& kf2) { return kf1->kf_id_ < kf2->kf_id_; }
    // bool operator<(KeyFrame* const& keyfrm) { std::cout << "< operator kf" << std::endl; return kf_id_ < keyfrm->kf_id_; }
    const auto& get_graph_node() { return *graph_node_; }

    void prepare_for_erasing(SlamReconstruction& reconstruction);
    void set_cannot_be_erased(bool flag) { cannot_be_erased_ = flag; }

public:
    size_t kf_id_;
    const size_t src_frm_id_;
    const std::string im_name_;

    //! keypoints of monocular or stereo left image
    const std::vector<cv::KeyPoint> keypts_;
    //! undistorted keypoints of monocular or stereo left image
    const std::vector<cv::KeyPoint> undist_keypts_;
    //! bearing vectors
    const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> bearings_;
    //! descriptors
    const cv::Mat descriptors_;
    std::vector<Landmark*> landmarks_;
    
    //! keypoint indices in each of the cells
    const std::vector<std::vector<std::vector<unsigned int>>> keypts_indices_in_cells_;
    const size_t num_keypts_;

    bool will_be_erased_ = false;
    bool cannot_be_erased_ = false;


    // ORB scale pyramid information
    //! number of scale levels
    size_t num_scale_levels_;
    //! scale factor
    float scale_factor_;
    //! log scale factor
    float log_scale_factor_;
    //! list of scale factors
    const std::vector<float> scale_factors_;
    //! list of inverse of scale factors
    const std::vector<float> inv_scale_factors_;
    //! list of sigma^2 (sigma=1.0 at scale=0) for optimization
    const std::vector<float> level_sigma_sq_;
    //! list of 1 / sigma^2 for optimization
    const std::vector<float> inv_level_sigma_sq_;
    //! identifier for local map update
    size_t local_map_update_identifier = 0;

    //! graph node
    const std::unique_ptr<openvslam::data::graph_node> graph_node_ = nullptr;


    private:
    Eigen::Matrix4f T_wc; // camera to world transformation, pose
    Eigen::Matrix4f T_cw; // world to camera transformation
    Eigen::Vector3f cam_center_;
};

struct KeyFrameCompare
{
    bool operator()(KeyFrame* kf1, KeyFrame* kf2) { return kf1->kf_id_ < kf2->kf_id_; }
    bool operator()(KeyFrame const* kf1, KeyFrame const * kf2) const { return kf1->kf_id_ < kf2->kf_id_; }
};

}