#pragma once
#include <map/map.h>
#include <Eigen/Eigen>

using namespace map;
namespace slam
{
class SlamMap : public map::Map
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // void InsertNewKeyFrame(Shot* shot);
    void UpdateLandmarksAfterKfInsert(map::Shot *shot);
    void RemoveRedundantLandmarks(const ShotId cur_keyfrm_id)
    {
        // TODO: Implement
    }
    void RemoveRedundantKeyFrames(Shot* cur_keyfrm, const ShotId origin_kf_id)
    {
        // TODO: Implement
    }
private:
    // std::unordered_map<ShotId, Shot*> keyframes_;
    std::vector<Landmark*> fresh_landmarks_;
};
}