#include "slam_reconstruction.h"
#include "landmark.h"
#include "keyframe.h"
#include "frame.h"

namespace cslam
{

KeyFrame* 
SlamReconstruction::create_new_keyframe(const Frame& frame)
{
    auto new_kf = new KeyFrame(next_kf_id, frame);
    ++next_kf_id;
    add_keyframe(new_kf);
    return new_kf;
}

Landmark* 
SlamReconstruction::create_new_landmark(KeyFrame* ref_kf, const Eigen::Vector3f& pos_w)
{
    auto new_lm = new Landmark(next_lm_id, ref_kf, pos_w);
    // std::cout << "Creating landmark " << next_lm_id << ", " << new_lm << std::endl;
    ++next_lm_id;
    add_landmark(new_lm);
    return new_lm;
}

std::vector<Landmark*> 
SlamReconstruction::get_all_landmarks() const {
    std::vector<Landmark*> landmarks;
    landmarks.reserve(landmarks_.size());
    for (const auto id_landmark : landmarks_) {
        landmarks.push_back(id_landmark.second);
    }
    return landmarks;
}

std::vector<KeyFrame*> 
SlamReconstruction::get_all_keyframes() const {
    std::vector<KeyFrame*> keyframes;
    keyframes.reserve(keyframes_.size());
    for (const auto id_keyframe : keyframes_) {
        keyframes.push_back(id_keyframe.second);
    }
    return keyframes;
}

void SlamReconstruction::add_landmark(Landmark* lm) {

    landmarks_[lm->lm_id_] = lm;
    // landmarks_.at(lm->lm_id_) = lm;
}

void SlamReconstruction::erase_landmark(Landmark* lm) {
    // auto lm = landmarks_[lm->lm_id_];
    // std::cout << "erase_landmark lm: " << lm << "/" << lm->lm_id_ << std::endl;
    landmarks_.erase(lm->lm_id_);
    //The problem with the delete is that the replace then doesn't work!
    // delete lm;
    // TODO: 実体を削除
}

void
SlamReconstruction::scale_map(KeyFrame* init_kf, KeyFrame* curr_kf, const double scale) const
{
    Eigen::Matrix4f cam_pose_cw = curr_kf->get_Tcw();
    cam_pose_cw.block<3, 1>(0, 3) *= scale;
    curr_kf->set_Tcw(cam_pose_cw);

    // scaling landmarks
    const auto landmarks = init_kf->landmarks_; //get_landmarks();
    for (auto lm : landmarks) {
        if (!lm) {
            continue;
        }
        lm->set_pos_in_world(lm->get_pos_in_world() * scale);
    }
}

// void
// SlamReconstruction::erase_keyframe(KeyFrame* kf)
// {
//     //maybe check for nullptr!
//     keyframes_.erase(kf->kf_id_);
// }

void
SlamReconstruction::apply_landmark_replace(Frame& frame)
{
    for (size_t idx = 0; idx < frame.num_keypts_; ++idx)
    {
        auto lm = frame.landmarks_.at(idx);
        if (lm == nullptr) continue;
        auto replaced_lm = lm->get_replaced();
        if (replaced_lm != nullptr)
        {
            // std::cout << "replacing: " << lm << " with " << replaced_lm << std::endl;
            frame.landmarks_.at(idx) = replaced_lm;
        }
    }
}

void
SlamReconstruction::add_keyframe(KeyFrame* keyfrm) {
    keyframes_[keyfrm->kf_id_] = keyfrm;
    // keyframes_.at(keyfrm->kf_id_) = keyfrm;
    // if (keyfrm->kf_id_ > max_keyfrm_id_) {
    //     max_keyfrm_id_ = keyfrm->kf_id_;
    // }
}

void SlamReconstruction::erase_keyframe(KeyFrame* keyfrm) {
    keyframes_.erase(keyfrm->kf_id_);
    delete keyfrm;
    // TODO: 実体を削除
}
}