#pragma once
#include <vector>
#include <unordered_set>

namespace cslam
{
class Landmark;
class KeyFrame;
class BrownPerspectiveCamera;
class GuidedMatcher;
class SlamReconstruction;
class LocalMapCleaner
{
public:
    LocalMapCleaner() = delete;
    LocalMapCleaner(const GuidedMatcher& guided_matcher, SlamReconstruction* map_db); //, BrownPerspectiveCamera* camera);
    void add_landmark(Landmark* new_lm)
    {
        fresh_landmarks_.push_back(new_lm);
    }
    // in OpenVSLAM == store_new_keyframe
    void update_lms_after_kf_insert(KeyFrame* new_kf);
    size_t remove_redundant_landmarks(const size_t cur_keyfrm_id);
    //TODO: Implement
    size_t remove_redundant_keyframes(KeyFrame* new_kf, const size_t origin_kf_id);
    static void count_redundant_observations(KeyFrame* keyfrm, size_t& num_valid_obs, size_t& num_redundant_obs); //const;
    // void update_new_keyframe(KeyFrame* curr_kf) const;
    void fuse_landmark_duplication(KeyFrame* curr_kf, const std::vector<KeyFrame*>& fuse_tgt_keyfrms) const;
private:
    std::vector<Landmark*> fresh_landmarks_;
    const GuidedMatcher& guided_matcher_;
    const BrownPerspectiveCamera& camera_;
    SlamReconstruction* map_db_;

};   
}