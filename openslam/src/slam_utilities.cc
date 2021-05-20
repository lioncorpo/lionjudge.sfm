#include "slam_utilities.h"
#include "slam_datastructures/keyframe.h"
#include "slam_datastructures/landmark.h"
#include "slam_datastructures/frame.h"


namespace cslam
{
void
SlamUtilities::update_new_keyframe(KeyFrame& curr_kf)
{
    // update the geometries
    const auto& cur_landmarks = curr_kf.landmarks_;
    for (const auto lm : cur_landmarks) {
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }
        lm->compute_descriptor();
        lm->update_normal_and_depth();
    }
}
// std::unordered_set<KeyFrame*>
std::vector<KeyFrame*>
SlamUtilities::get_second_order_covisibilities_for_kf(const KeyFrame* kf, const size_t first_order_thr, const size_t second_order_thr)
{
    const auto cur_covisibilities = kf->graph_node_->get_top_n_covisibilities(first_order_thr);

    std::unordered_set<KeyFrame*> fuse_tgt_keyfrms;
    fuse_tgt_keyfrms.reserve(cur_covisibilities.size() * 2);

    for (const auto first_order_covis : cur_covisibilities) {
        if (first_order_covis->will_be_erased()) {
            continue;
        }

        // check if the keyframe is aleady inserted
        if (static_cast<bool>(fuse_tgt_keyfrms.count(first_order_covis))) {
            continue;
        }

        fuse_tgt_keyfrms.insert(first_order_covis);
        std::cout << "fuse_tgt_keyfrms.insert(first_order_covis): " << first_order_covis->im_name_ << std::endl;

        // get the covisibilities of the covisibility of the current keyframe
        const auto ngh_covisibilities = first_order_covis->graph_node_->get_top_n_covisibilities(second_order_thr);
        for (const auto second_order_covis : ngh_covisibilities) {
            if (second_order_covis->will_be_erased()) {
                continue;
            }
            // "the covisibilities of the covisibility" contains the current keyframe
            if (*second_order_covis == *kf) {
                continue;
            }

            fuse_tgt_keyfrms.insert(second_order_covis);
            std::cout << "fuse_tgt_keyfrms.insert(second_order_covis): " << second_order_covis->im_name_ << std::endl;
        }
    }

    for (const auto& frm: fuse_tgt_keyfrms)
    {
        std::cout << "frm: " << frm->im_name_ << std::endl;
    }
    // return fuse_tgt_keyfrms;
    //TODO: this copy is unnecessary and only used to keep the order
    std::vector<KeyFrame*> fuse_tgt_keyfrms_vec(fuse_tgt_keyfrms.cbegin(), fuse_tgt_keyfrms.cend());
    // std::copy(fuse_tgt_keyfrms.cbegin(), fuse_tgt_keyfrms.cbegin(), )
    return fuse_tgt_keyfrms_vec;
    
}

std::vector<Landmark*>
SlamUtilities::update_local_landmarks(const std::vector<KeyFrame*>& local_keyframes, const size_t curr_frm_id)
{
    std::vector<Landmark*> local_landmarks;
    for (auto keyframe : local_keyframes)
    {
        for (auto lm : keyframe->landmarks_)
        {
            if (lm == nullptr) continue;
            // do not add twice
            if (lm->identifier_in_local_map_update_ == curr_frm_id) continue;
            lm->identifier_in_local_map_update_ = curr_frm_id;
            local_landmarks.push_back(lm);
        }
    }
    return local_landmarks;
}

std::vector<KeyFrame*>
SlamUtilities::update_local_keyframes(Frame& frame) 
{
    constexpr unsigned int max_num_local_keyfrms{60};

    // count the number of sharing landmarks between the current frame and each of the neighbor keyframes
    // key: keyframe, value: number of sharing landmarks
    std::unordered_map<KeyFrame*, unsigned int> keyfrm_weights;
    for (unsigned int idx = 0; idx < frame.num_keypts_; ++idx) {
        auto lm = frame.landmarks_.at(idx);
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            // std::cout << "lm->will_be_erased()" << std::endl;
            // exit(0);
            // kf.landmarks_.at(idx) = nullptr;
            //TODO: Write this maybe in a clean-up function!
            frame.landmarks_.at(idx) = nullptr;
            continue;
        }

        const auto observations = lm->get_observations();
        for (auto obs : observations) {
            ++keyfrm_weights[obs.first];
        }
    }

    if (keyfrm_weights.empty()) {
        return std::vector<KeyFrame*>();
    }

    // set the aforementioned keyframes as local keyframes
    // and find the nearest keyframe
    unsigned int max_weight = 0;
    KeyFrame* nearest_covisibility = nullptr;

    std::vector<KeyFrame*> local_keyfrms_; //.clear();
    local_keyfrms_.reserve(4 * keyfrm_weights.size());

    for (auto& keyfrm_weight : keyfrm_weights) {
        auto keyfrm = keyfrm_weight.first;
        const auto weight = keyfrm_weight.second;

        if (keyfrm->will_be_erased()) {
            continue;
        }

        local_keyfrms_.push_back(keyfrm);

        // avoid duplication
        keyfrm->local_map_update_identifier = frame.frame_id;

        // update the nearest keyframe
        if (max_weight < weight) {
            max_weight = weight;
            nearest_covisibility = keyfrm;
        }
    }

    // add the second-order keyframes to the local landmarks
    auto add_local_keyframe = [&](KeyFrame* keyfrm) {
        if (!keyfrm) {
            return false;
        }
        if (keyfrm->will_be_erased()) {
            return false;
        }
        // avoid duplication
        if (keyfrm->local_map_update_identifier == frame.frame_id) {
            return false;
        }
        keyfrm->local_map_update_identifier = frame.frame_id;
        local_keyfrms_.push_back(keyfrm);
        return true;
    };
    for (auto iter = local_keyfrms_.cbegin(), end = local_keyfrms_.cend(); iter != end; ++iter) {
        if (max_num_local_keyfrms < local_keyfrms_.size()) {
            break;
        }

        auto keyfrm = *iter;

        // covisibilities of the neighbor keyframe
        const auto neighbors = keyfrm->graph_node_->get_top_n_covisibilities(10);
        for (auto neighbor : neighbors) {
            if (add_local_keyframe(neighbor)) {
                break;
            }
        }

        // children of the spanning tree
        const auto spanning_children = keyfrm->graph_node_->get_spanning_children();
        for (auto child : spanning_children) {
            if (add_local_keyframe(child)) {
                break;
            }
        }

        // parent of the spanning tree
        auto parent = keyfrm->graph_node_->get_spanning_parent();
        add_local_keyframe(parent);
    }

    // update the reference keyframe with the nearest one
    // if (nearest_covisibility) {
    //     ref_keyfrm_ = nearest_covisibility;
    //     curr_frm_.ref_keyfrm_ = ref_keyfrm_;
    // }
    return local_keyfrms_;
}

}