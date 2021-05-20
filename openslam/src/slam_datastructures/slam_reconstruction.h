#pragma once
#include <vector>
#include <unordered_map>
#include <Eigen/Eigen>
namespace cslam
{
class KeyFrame;
class Frame;
class Landmark;
class SlamReconstruction
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    /**
     * Erase keyframe from the database
     * @param keyfrm
     */
    void erase_keyframe(KeyFrame* kf);
    /**
     * Erase landmark from the database
     * @param lm
     */
    void erase_landmark(Landmark* lm);
    /**
     * Get all of the keyframes in the database
     * @return
     */
    std::vector<KeyFrame*> get_all_keyframes() const;
    /**
     * Get all of the landmarks in the database
     * @return
     */
    std::vector<Landmark*> get_all_landmarks() const;

    /**
     * Get the number of landmarks
     * @return
     */
    size_t get_num_landmarks() const { return landmarks_.size(); }
    // /**
    //  * Get the maximum keyframe ID
    //  * @return
    //  */
    // size_t get_max_keyframe_id() const { return 
    // max_keyfrm_id_; }

    size_t get_num_keyframes() const { return keyframes_.size(); }

    // KeyFrame* create_new_keyframe(const size_t kf_id, const Frame& frame);
    // Landmark* create_new_landmark(const size_t lm_id, KeyFrame* ref_kf, const Eigen::Vector3f& pos_w);
    KeyFrame* create_new_keyframe(const Frame& frame);
    Landmark* create_new_landmark(KeyFrame* ref_kf, const Eigen::Vector3f& pos_w);

    void apply_landmark_replace(Frame& frame);
    void scale_map(KeyFrame* kf1, KeyFrame* kf2, const double scale) const;
private:
    //! IDs and keyframes
    std::unordered_map<unsigned int, KeyFrame*> keyframes_;
    //! IDs and landmarks
    std::unordered_map<unsigned int, Landmark*> landmarks_;
    size_t next_kf_id = 0;
    size_t next_lm_id = 0;
     /**
     * Add landmark to the database
     * @param lm
     */
    void add_landmark(Landmark* lm);
    /**
     * Add keyframe to the database
     * @param keyfrm
     */
    void add_keyframe(KeyFrame* keyfrm);

    // TODO: implement
    // erase landmark with "surroundings"
    // erase observation with "surroundings"
    // erase keyframe with "surroundings"

};
    
}