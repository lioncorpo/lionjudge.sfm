#include "landmark.h"
#include "keyframe.h"
#include "frame.h"
#include "third_party/openvslam/util/guided_matching.h"
#include "slam_reconstruction.h"
#include <iostream>
namespace cslam
{

Landmark::Landmark(const size_t lm_id, KeyFrame* ref_kf, const Eigen::Vector3f& pos_w):
    lm_id_(lm_id), ref_keyfrm_(ref_kf), ref_kf_id_(ref_kf->kf_id_), pos_w_(pos_w)
    {
        scale_level_in_tracking_ = 0;
    };
void 
Landmark::add_observation(KeyFrame* keyfrm, size_t idx) {
    if (observations_.count(keyfrm)) {
        return;
    }
    // std::cout << "add obs: " << keyfrm << ", " << keyfrm->kf_id_ << " idx: " << idx <<  "lm: " << lm_id_ << " ptr: " << this << std::endl;
    observations_[keyfrm] = idx;
    num_observations_ += 1;
}

void 
Landmark::erase_observation(KeyFrame* keyfrm, SlamReconstruction* reconstruction) {
    if (observations_.count(keyfrm)) {
        // int idx = observations_.at(keyfrm);
        // if (0 <= keyfrm->stereo_x_right_.at(idx)) {
        //     num_observations_ -= 2;
        // }
        // else {
        num_observations_ -= 1;
        // }

        observations_.erase(keyfrm);

        if (ref_keyfrm_ == keyfrm) {
            ref_keyfrm_ = observations_.begin()->first;
        }

        // If only 2 observations or less, discard point
        if (num_observations_ <= 2) {
            prepare_for_erasing();
            reconstruction->erase_landmark(this);
        }
    }
}

size_t 
Landmark::predict_scale_level(const float cam_to_lm_dist, const float log_scale_factor, const size_t num_scale_levels) const 
{
    const auto ratio = max_valid_dist_ / cam_to_lm_dist;

    const auto pred_scale_level = static_cast<int>(std::ceil(std::log(ratio) / log_scale_factor));
    if (pred_scale_level < 0) {
        return 0;
    }
    else if (num_scale_levels <= static_cast<unsigned int>(pred_scale_level)) {
        return num_scale_levels - 1;
    }
    else {
        return static_cast<unsigned int>(pred_scale_level);
    }
    //Should be the same as:
    // if (pred_scale_level < 0) return 0;
    // if (num_scale_levels_ <= static_cast<unsigned int>(pred_scale_level)) return num_scale_levels_ - 1;
    // return static_cast<unsigned int>(pred_scale_level);
}

size_t 
Landmark::predict_scale_level(const float cam_to_lm_dist, const KeyFrame& keyfrm) const 
{
    return predict_scale_level(cam_to_lm_dist, keyfrm.log_scale_factor_, keyfrm.num_scale_levels_);
}

size_t 
Landmark::predict_scale_level(const float cam_to_lm_dist, const Frame& frm) const 
{
    return predict_scale_level(cam_to_lm_dist, frm.log_scale_factor_, frm.num_scale_levels_);
}

void 
Landmark::prepare_for_erasing() 
{
    // std::cout << "prepare_for_erasing: " << this << ", " << lm_id_ << std::endl;
    for (const auto& keyfrm_and_idx : observations_) {
        keyfrm_and_idx.first->erase_landmark_with_index(keyfrm_and_idx.second);
    }
    observations_.clear();
    will_be_erased_ = true;
    // map_db_->erase_landmark(this);
}

void 
Landmark::replace(Landmark* lm) {
    if (lm->lm_id_ == this->lm_id_) {
        return;
    }
    // std::cout << "Replace " << this << " with " << lm << std::endl;
    unsigned int num_observable, num_observed;
    //TODO: Probably move the observations.clear() to the end and avoid copying
    std::map<KeyFrame*, size_t, KeyFrameCompare> observations;
    observations = observations_;
    observations_.clear();
    will_be_erased_ = true;
    num_observable = num_observable_;
    num_observed = num_observed_;
    replaced_ = lm;
    // std::cout << "replaced lm = " << replaced_ << std::endl;

    for (const auto& keyfrm_and_idx : observations) {
        KeyFrame* keyfrm = keyfrm_and_idx.first;

        if (!lm->is_observed_in_keyframe(keyfrm)) {
            keyfrm->replace_landmark(lm, keyfrm_and_idx.second);
            lm->add_observation(keyfrm, keyfrm_and_idx.second);
        }
        else {
            keyfrm->erase_landmark_with_index(keyfrm_and_idx.second);
        }
    }

    lm->increase_num_observed(num_observed);
    lm->increase_num_observable(num_observable);
    lm->compute_descriptor();

    // map_db_->erase_landmark(this);
}


void
Landmark::compute_descriptor()
{
    if (observations_.empty()) return;
    std::cout << "compute_descriptor: " << lm_id_ << std::endl;

    // 対応している特徴点の特徴量を集める
    std::vector<cv::Mat> descriptors;
    descriptors.reserve(observations_.size());
    for (const auto& observation : observations_) {
        auto keyfrm = observation.first;
        const auto idx = observation.second;

        // if (!keyfrm->will_be_erased()) {
            descriptors.push_back(keyfrm->descriptors_.row(idx));
            std::cout << "keyfrm->descriptors_.row(" << idx << "): " << keyfrm->descriptors_.row(idx) << std::endl;
        // }
    }

    // ハミング距離の中央値を計算
    // Calculate median Hamming distance

    // まず特徴量間の距離を全組み合わせで計算
    // First, calculate the distance between features in all combinations
    const auto num_descs = descriptors.size();
    // std::cout << "Computing: " << num_descs << std::endl;
    std::vector<std::vector<unsigned int>> hamm_dists(num_descs, std::vector<unsigned int>(num_descs));
    for (unsigned int i = 0; i < num_descs; ++i) {
        hamm_dists.at(i).at(i) = 0;
        for (unsigned int j = i + 1; j < num_descs; ++j) {
            const auto dist = GuidedMatcher::compute_descriptor_distance_32(descriptors.at(i), descriptors.at(j));
            hamm_dists.at(i).at(j) = dist;
            hamm_dists.at(j).at(i) = dist;
            std::cout << "i/j: " << i << "/" << j << " dist: " << dist << std::endl;
        }
    }

    // 中央値に最も近いものを求める
    // Find the closest to the median
    unsigned int best_median_dist = GuidedMatcher::MAX_HAMMING_DIST;
    unsigned int best_idx = 0;
    for (unsigned idx = 0; idx < num_descs; ++idx) {
        std::vector<unsigned int> partial_hamm_dists(hamm_dists.at(idx).begin(), hamm_dists.at(idx).begin() + num_descs);
        std::sort(partial_hamm_dists.begin(), partial_hamm_dists.end());
        const auto median_dist = partial_hamm_dists.at(static_cast<unsigned int>(0.5 * (num_descs - 1)));
        std::cout << "median_dist: " << median_dist << "num_descs: " << num_descs << std::endl;
        if (median_dist < best_median_dist) {
            best_median_dist = median_dist;
            best_idx = idx;
        }
    }
    std::cout << "Final descriptor at " << best_idx << " desc: " << descriptors.at(best_idx);
    descriptor_ = descriptors.at(best_idx).clone();
}
void 
Landmark::update_normal_and_depth()
{
    if (observations_.empty()) {
        return;
    }

    Eigen::Vector3f mean_normal = Eigen::Vector3f::Zero();
    unsigned int num_observations = 0;
    for (const auto& observation : observations_) {
        auto keyfrm = observation.first;
        const Eigen::Vector3f cam_center = keyfrm->get_cam_center();
        const Eigen::Vector3f normal = pos_w_ - cam_center;
        mean_normal = mean_normal + normal / normal.norm();
        ++num_observations;
    }
    //TODO: num_obs == observations.size()
    const Eigen::Vector3f cam_to_lm_vec = pos_w_ - ref_keyfrm_->get_cam_center();
    const auto dist = cam_to_lm_vec.norm();
    const auto scale_level = ref_keyfrm_->undist_keypts_.at(observations_.at(ref_keyfrm_)).octave;
    const auto scale_factor = ref_keyfrm_->scale_factors_.at(scale_level);
    const auto num_scale_levels = ref_keyfrm_->num_scale_levels_;

    max_valid_dist_ = dist * scale_factor;
    min_valid_dist_ = max_valid_dist_ / ref_keyfrm_->scale_factors_.at(num_scale_levels - 1);
    mean_normal_ = mean_normal / num_observations;
}

}