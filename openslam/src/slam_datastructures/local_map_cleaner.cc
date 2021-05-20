#include "local_map_cleaner.h"
#include "keyframe.h"
#include "landmark.h"
#include "camera.h"
#include "third_party/openvslam/util/guided_matching.h"
#include "slam_reconstruction.h"
namespace cslam
{

LocalMapCleaner::LocalMapCleaner(const GuidedMatcher& guided_matcher, SlamReconstruction* map_db): //, BrownPerspectiveCamera* camera):
    guided_matcher_(guided_matcher), camera_(guided_matcher.camera_), map_db_(map_db)
{

}

void 
LocalMapCleaner::update_lms_after_kf_insert(KeyFrame* new_kf)
{
    // landmarks_
    for (unsigned int idx = 0; idx < new_kf->landmarks_.size(); ++idx) {
        auto lm = new_kf->landmarks_.at(idx);
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }

        // if `lm` does not have the observation information from `cur_keyfrm_`,
        // add the association between the keyframe and the landmark
        if (lm->is_observed_in_keyframe(new_kf)) {
            // if `lm` is correctly observed, make it be checked by the local map cleaner
            // local_map_cleaner_->add_fresh_landmark(lm);
            fresh_landmarks_.push_back(lm);
            continue;
        }

        // update connection
        lm->add_observation(new_kf, idx);
        // update geometry
        lm->update_normal_and_depth();
        lm->compute_descriptor();
    }
    // std::cout << "Before update connections!" << std::endl;
    new_kf->graph_node_->update_connections();
    // std::cout << "AFter update connections!" << std::endl;
    const auto cov = new_kf->graph_node_->get_top_n_covisibilities(10);
    // std::cout<< "cov: " << cov.size() << std::endl;
}

void
LocalMapCleaner::fuse_landmark_duplication(KeyFrame* curr_kf, const std::vector<KeyFrame*>& fuse_tgt_keyfrms) const
{
    auto cur_landmarks = curr_kf->landmarks_;
    // std::cout << "Before first for" <<std::endl;

    // go through kfs and fuse!
    // reproject the landmarks observed in the current keyframe to each of the targets, and acquire
    // - additional matches
    // - duplication of matches
    // then, add matches and solve duplication
    for (const auto fuse_tgt_keyfrm : fuse_tgt_keyfrms) {
            const auto n_fused = guided_matcher_.replace_duplication(fuse_tgt_keyfrm, cur_landmarks);
            std::cout << "Fused: " << n_fused << " for " << fuse_tgt_keyfrm->im_name_ << std::endl;
    }
    // std::cout << "After first for" <<std::endl;
    // reproject the landmarks observed in each of the targets to each of the current frame, and acquire
    // - additional matches
    // - duplication of matches
    // then, add matches and solve duplication
    std::unordered_set<Landmark*> candidate_landmarks_to_fuse;
    candidate_landmarks_to_fuse.reserve(fuse_tgt_keyfrms.size() * curr_kf->landmarks_.size());

    for (const auto fuse_tgt_keyfrm : fuse_tgt_keyfrms) {
        const auto fuse_tgt_landmarks = fuse_tgt_keyfrm->landmarks_;//fuse_tgt_keyfrm->get_landmarks();

        for (const auto lm : fuse_tgt_landmarks) {
            if (!lm) {
                continue;
            }
            if (lm->will_be_erased()) {
                continue;
            }

            if (static_cast<bool>(candidate_landmarks_to_fuse.count(lm))) {
                continue;
            }
            candidate_landmarks_to_fuse.insert(lm);
        }
    }
    std::cout << "After second for" <<std::endl;
    const auto n_fused = guided_matcher_.replace_duplication(curr_kf, candidate_landmarks_to_fuse);
    std::cout << "Fused: " << n_fused << " for curr" << curr_kf->im_name_ << std::endl;

}

void 
LocalMapCleaner::count_redundant_observations(KeyFrame* keyfrm, size_t& num_valid_obs, size_t& num_redundant_obs) //const 
{
    // if the number of keyframes that observes the landmark with more reliable scale than the specified keyframe does,
    // it is considered as redundant
    constexpr size_t num_better_obs_thr{3};

    num_valid_obs = 0;
    num_redundant_obs = 0;

    const auto landmarks = keyfrm->landmarks_;//->get_landmarks();
    for (size_t idx = 0; idx < landmarks.size(); ++idx) {
        auto lm = landmarks.at(idx);
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }

        // if depth is within the valid range, it won't be considered
        // const auto depth = keyfrm->depths_.at(idx);
        // if (!is_monocular_ && (depth < 0.0 || keyfrm->depth_thr_ < depth)) {
        //     continue;
        // }

        ++num_valid_obs;

        // if the number of the obs is smaller than the threshold, cannot remote the observers
        if (lm->num_observations() <= num_better_obs_thr) {
            continue;
        }

        // `keyfrm` observes `lm` with the scale level `scale_level`
        const auto scale_level = keyfrm->undist_keypts_.at(idx).octave;
        // get observers of `lm`
        const auto observations = lm->get_observations();

        bool obs_by_keyfrm_is_redundant = false;

        // the number of the keyframes that observe `lm` with the more reliable (closer) scale
        size_t num_better_obs = 0;

        for (const auto obs : observations) {
            const auto ngh_keyfrm = obs.first;
            if (*ngh_keyfrm == *keyfrm) {
                continue;
            }

            // `ngh_keyfrm` observes `lm` with the scale level `ngh_scale_level`
            const auto ngh_scale_level = ngh_keyfrm->undist_keypts_.at(obs.second).octave;

            // compare the scale levels
            if (ngh_scale_level <= scale_level + 1) {
                // the observation by `ngh_keyfrm` is more reliable than `keyfrm`
                ++num_better_obs;
                if (num_better_obs_thr <= num_better_obs) {
                    // if the number of the better observations is greater than the threshold,
                    // consider the observation of `lm` by `keyfrm` is redundant
                    obs_by_keyfrm_is_redundant = true;
                    break;
                }
            }
        }

        if (obs_by_keyfrm_is_redundant) {
            ++num_redundant_obs;
        }
    }
}

size_t
LocalMapCleaner::remove_redundant_landmarks(const size_t cur_keyfrm_id) {
    constexpr float observed_ratio_thr = 0.3;
    constexpr unsigned int num_reliable_keyfrms = 2;
    // const unsigned int num_obs_thr = is_monocular_ ? 2 : 3;
    constexpr size_t num_obs_thr{2};

    // states of observed landmarks
    enum class lm_state_t { Valid, Invalid, NotClear };

    unsigned int num_removed = 0;
    auto iter = fresh_landmarks_.begin();
    while (iter != fresh_landmarks_.end()) {
        auto lm = *iter;

        // decide the state of lms the buffer
        auto lm_state = lm_state_t::NotClear;
        if (lm->will_be_erased()) {
            // in case `lm` will be erased
            // remove `lm` from the buffer
            lm_state = lm_state_t::Valid;
        }
        else if (lm->get_observed_ratio() < observed_ratio_thr) {
            // if `lm` is not reliable
            // remove `lm` from the buffer and the database
            lm_state = lm_state_t::Invalid;
        }
        else if (num_reliable_keyfrms + lm->ref_kf_id_ <= cur_keyfrm_id
                 && lm->num_observations() <= num_obs_thr) {
            // if the number of the observers of `lm` is small after some keyframes were inserted
            // remove `lm` from the buffer and the database
            lm_state = lm_state_t::Invalid;
        }
        else if (num_reliable_keyfrms + 1U + lm->ref_kf_id_ <= cur_keyfrm_id) {
            // if the number of the observers of `lm` is sufficient after some keyframes were inserted
            // remove `lm` from the buffer
            lm_state = lm_state_t::Valid;
        }

        // select to remove `lm` according to the state
        if (lm_state == lm_state_t::Valid) {
            iter = fresh_landmarks_.erase(iter);
        }
        else if (lm_state == lm_state_t::Invalid) {
            ++num_removed;
            lm->prepare_for_erasing();
            iter = fresh_landmarks_.erase(iter);
            map_db_->erase_landmark(lm);
        }
        else {
            // hold decision because the state is NotClear
            iter++;
        }
    }
    std::cout << "remove_redundant_landmarks: " << num_removed << std::endl;
    return num_removed;
}


size_t
LocalMapCleaner::remove_redundant_keyframes(KeyFrame* cur_keyfrm, const size_t origin_kf_id) 
{
    // window size not to remove
    constexpr unsigned int window_size_not_to_remove = 2;
    // if the redundancy ratio of observations is larger than this threshold,
    // the corresponding keyframe will be erased
    constexpr float redundant_obs_ratio_thr = 0.9;

    size_t num_removed = 0;
    // check redundancy for each of the covisibilities
    const auto cur_covisibilities = cur_keyfrm->graph_node_->get_covisibilities();
    for (const auto covisibility : cur_covisibilities) {
        // cannot remove the origin
        if (covisibility->kf_id_ == origin_kf_id) {
            continue;
        }
        // cannot remove the recent keyframe(s)
        if (covisibility->kf_id_ <= cur_keyfrm->kf_id_
            && cur_keyfrm->kf_id_ <= covisibility->kf_id_ + window_size_not_to_remove) {
            continue;
        }

        // count the number of redundant observations (num_redundant_obs) and valid observations (num_valid_obs)
        // for the covisibility
        size_t num_redundant_obs{0}, num_valid_obs{0};
        // unsigned int num_valid_obs = 0;
        count_redundant_observations(covisibility, num_valid_obs, num_redundant_obs);
        std::cout << covisibility->im_name_ << " num_redundant_obs" << num_redundant_obs 
                  << "num_valid_obs" << num_valid_obs << " factor: " << static_cast<float>(num_redundant_obs) / num_valid_obs << std::endl;
        // if the redundant observation ratio of `covisibility` is larger than the threshold, it will be removed
        if (redundant_obs_ratio_thr <= static_cast<float>(num_redundant_obs) / num_valid_obs) {
            ++num_removed;
            covisibility->prepare_for_erasing(*map_db_);
        }
    }

    return num_removed;
}

}