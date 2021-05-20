#include "frame.h"
#include "landmark.h"
// #include "types.h"
// #include <pybind11/pybind11.h>
#include "third_party/openvslam/feature/orb_extractor.h"
// #include "camera.h"
namespace cslam
{
Frame::Frame(const csfm::pyarray_uint8 image, const csfm::pyarray_uint8 mask,
             const std::string& img_name, const size_t frame_id, 
             openvslam::feature::orb_extractor* orb_extractor):
    im_name(img_name), frame_id(frame_id), //scale_factors_(orb_extractor->get_scale_factors()),
    T_cw(Eigen::Matrix4f::Identity()), T_wc(Eigen::Matrix4f::Identity()), cam_center_(Eigen::Vector3f::Zero())
{
    orb_extractor->extract_orb_py2(image, mask, *this);
    update_orb_info(orb_extractor);
    num_keypts_ = keypts_.size();
    // std::cout << "Allocating " << num_keypts_ << " for frame: " << img_name << std::endl;
    landmarks_ = std::vector<Landmark*>(num_keypts_, nullptr);
    outlier_flags_ = std::vector<bool>(num_keypts_, false);
    mParentKf = nullptr;
}

 void 
 Frame::update_orb_info(openvslam::feature::orb_extractor* orb_extractor)
 {
    num_scale_levels_ = orb_extractor->get_num_scale_levels();
    scale_factor_ = orb_extractor->get_scale_factor();
    log_scale_factor_ = std::log(scale_factor_);
    scale_factors_ = orb_extractor->get_scale_factors();
    inv_scale_factors_ = orb_extractor->get_inv_scale_factors();
    level_sigma_sq_ = orb_extractor->get_level_sigma_sq();
    inv_level_sigma_sq_ = orb_extractor->get_inv_level_sigma_sq();
 }

void 
Frame::add_landmark(Landmark* lm, size_t idx)
{
    // landmarks_[idx] = lm;
    // outlier_flags_[idx] = false;
    landmarks_.at(idx) = lm;
    outlier_flags_.at(idx) = false;
}

py::object
Frame::getKptsUndist() const
{
    // Convert to numpy.
    cv::Mat keys(undist_keypts_.size(), 5, CV_32F);
    for (int i = 0; i < (int) undist_keypts_.size(); ++i) {
        keys.at<float>(i, 0) = undist_keypts_[i].pt.x;
        keys.at<float>(i, 1) = undist_keypts_[i].pt.y;
        keys.at<float>(i, 2) = undist_keypts_[i].size;
        keys.at<float>(i, 3) = undist_keypts_[i].angle;
        keys.at<float>(i, 4) = undist_keypts_[i].octave;
    }
    return csfm::py_array_from_data(keys.ptr<float>(0), keys.rows, keys.cols);
}

py::object
Frame::getDescPy() const
{
    return csfm::py_array_from_data(descriptors_.ptr<unsigned char>(0), descriptors_.rows, descriptors_.cols);
}

py::object
Frame::getKptsPy() const
{
    // Convert to numpy.
    cv::Mat keys(keypts_.size(), 5, CV_32F);
    for (int i = 0; i < (int) keypts_.size(); ++i) {
        keys.at<float>(i, 0) = keypts_[i].pt.x;
        keys.at<float>(i, 1) = keypts_[i].pt.y;
        keys.at<float>(i, 2) = keypts_[i].size;
        keys.at<float>(i, 3) = keypts_[i].angle;
        keys.at<float>(i, 4) = keypts_[i].octave;
    }
    return csfm::py_array_from_data(keys.ptr<float>(0), keys.rows, keys.cols);
}

py::object
Frame::getKptsAndDescPy() const
{
    // Convert to numpy.
    cv::Mat keys(keypts_.size(), 5, CV_32F);
    for (int i = 0; i < (int) keypts_.size(); ++i) {
        keys.at<float>(i, 0) = keypts_[i].pt.x;
        keys.at<float>(i, 1) = keypts_[i].pt.y;
        keys.at<float>(i, 2) = keypts_[i].size;
        keys.at<float>(i, 3) = keypts_[i].angle;
        keys.at<float>(i, 4) = keypts_[i].octave;
    }

    py::list retn;
    retn.append(csfm::py_array_from_data(keys.ptr<float>(0), keys.rows, keys.cols));
    retn.append(csfm::py_array_from_data(descriptors_.ptr<unsigned char>(0), descriptors_.rows, descriptors_.cols));
    return retn;
}

// std::vector<Landmark*, Eigen::Vector2f> 
std::vector<Landmark*>
Frame::get_valid_lms()
{
    std::vector<Landmark*> landmark;
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> obs; //x, y, scale
    for (size_t i = 0; i < num_keypts_; ++i)//(auto lm : landmarks_)
    {
        auto lm = landmarks_[i];
        if (lm == nullptr) continue;
        landmark.push_back(lm);
        const auto kp = keypts_[i];
        obs.push_back(Eigen::Vector3f(kp.pt.x, kp.pt.y, kp.size));
    }
    return landmark;
}

std::vector<size_t> 
Frame::get_valid_idx() const
{
    std::vector<size_t> valid_idx;
    for (size_t i = 0; i < num_keypts_; ++i)
    {
        auto lm = landmarks_[i];
        if (lm == nullptr) continue;
        valid_idx.push_back(i);
    }
    return valid_idx;
}

void
Frame::set_outlier(const std::vector<size_t>& invalid_ids)
{
    for (const auto id : invalid_ids)
        outlier_flags_[id] = true;
}

size_t
Frame::clean_and_tick_landmarks() //const std::vector<size_t>& invalid_ids)
{
    size_t num_tracked_landmarks{0};
    for (size_t idx = 0; idx < landmarks_.size(); ++idx)
    {
        auto lm = landmarks_.at(idx);
        if (lm == nullptr)
            continue;
        // if (outlier_flags_[idx]) 
        if (outlier_flags_.at(idx))
        {
            // landmarks_[idx] = nullptr;
            landmarks_.at(idx) = nullptr;
        }
        else
        {
            lm->increase_num_observed();
            ++num_tracked_landmarks;
        }
        
    }
    return num_tracked_landmarks;
}

py::object
Frame::get_valid_keypts() const
{
    const auto n_valid_pts = num_keypts_ - std::count(landmarks_.cbegin(), landmarks_.cend(),nullptr);
    // Convert to numpy.
    cv::Mat keys(n_valid_pts, 3, CV_32F);
    size_t idx2{0};
    for (size_t i = 0; i < keypts_.size(); ++i) {
    if (landmarks_[i] != nullptr)
        {
            keys.at<float>(idx2, 0) = keypts_[i].pt.x;
            keys.at<float>(idx2, 1) = keypts_[i].pt.y;
            keys.at<float>(idx2, 2) = keypts_[i].size;
            idx2++;
        }
    }
    return csfm::py_array_from_data(keys.ptr<float>(0), keys.rows, keys.cols);
}

size_t
Frame::discard_outliers()
{
    size_t n_matches = 0;
    for (size_t idx = 0; idx < num_keypts_; ++idx)
    {
        //check lm
        auto lm = landmarks_.at(idx);
        if (lm != nullptr)
        {
            if (outlier_flags_.at(idx))
            {
                landmarks_.at(idx) = nullptr;
                outlier_flags_.at(idx) = false;
                lm->is_observable_in_tracking_ = false;
                lm->identifier_in_local_lm_search_ = frame_id;
            }
            else 
                ++n_matches;
        }
    }

    return n_matches;
}

}