#pragma once
#include <Eigen/Eigen>
#include <vector>
#include <opencv2/core.hpp>
#include <pybind11/pybind11.h>
#include "types.h"

namespace py = pybind11;

namespace openvslam{ namespace feature { class orb_extractor; }}

namespace cslam
{
class KeyFrame;
class Landmark;
// class BrownPerspectiveCamera;
class Frame
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Frame(const csfm::pyarray_uint8 image, const csfm::pyarray_uint8 mask, const std::string& img_name,
          const size_t frame_id, openvslam::feature::orb_extractor* orb_extractor);
    void add_landmark(Landmark* lm, size_t idx);
    py::object get_lm_and_obs(); //cslam::Frame& frame) const;
    const std::string im_name;
    const size_t frame_id;
    KeyFrame* mParentKf;
    std::vector<cv::KeyPoint> keypts_; // extracted keypoints
    std::vector<cv::KeyPoint> undist_keypts_; // undistorted keypoints
    cv::Mat descriptors_;
    std::vector<Landmark*> landmarks_;
    std::vector<bool> outlier_flags_;
    std::vector<std::vector<std::vector<unsigned int>>> keypts_indices_in_cells_;
    //! bearing vectors
    // Eigen::eigen_alloc_vector<Eigen::Vector3f> bearings_;
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> bearings_;
    py::object getKptsAndDescPy() const;
    py::object getKptsUndist() const;
    py::object getKptsPy() const;
    py::object getDescPy() const;
    py::object get_valid_keypts() const;
    std::vector<size_t> get_valid_idx() const;
    void set_outlier(const std::vector<size_t>& invalid_ids);
    size_t clean_and_tick_landmarks();
    // void tick_tracked_lms();
    std::vector<Landmark*> get_valid_lms();
    size_t num_keypts_;
    Eigen::Vector3f get_cam_center() const { return cam_center_; }
    Eigen::Matrix4f get_Twc() const { return T_wc; }
    Eigen::Matrix4f get_Tcw() const { return T_cw; }
    void set_Tcw(const Eigen::Matrix4f& T_cw_) { 
        T_cw = T_cw_;
        T_wc = T_cw.inverse();
        cam_center_ = T_wc.block<3,1>(0,3);
    }
    void set_Twc(const Eigen::Matrix4f& T_wc_)
    {
        T_wc = T_wc_;
        T_cw = T_wc.inverse();
        // cam_pose_cw_ = T_wc.inverse();
        // cam_center_ = T_cw.block<3,1>(0,3);//-rot_wc * trans_cw;
        cam_center_ = T_wc.block<3,1>(0,3);
    }
    size_t discard_outliers();
    // ORB scale pyramid information
    //! number of scale levels
    size_t num_scale_levels_;
    //! scale factor
    float scale_factor_;
    //! log scale factor
    float log_scale_factor_;
    //! list of scale factors
    std::vector<float> scale_factors_;
    //! list of inverse of scale factors
    std::vector<float> inv_scale_factors_;
    //! list of sigma^2 (sigma=1.0 at scale=0) for optimization
    std::vector<float> level_sigma_sq_;
    //! list of 1 / sigma^2 for optimization
    std::vector<float> inv_level_sigma_sq_;

    void update_orb_info(openvslam::feature::orb_extractor* orb_extractor);
private:
    Eigen::Matrix4f T_cw, T_wc;
    Eigen::Vector3f cam_center_;

};
}