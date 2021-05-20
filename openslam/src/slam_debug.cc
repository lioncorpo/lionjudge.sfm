#include <opencv2/features2d.hpp>
#include "slam_debug.h"
#include "slam_datastructures/frame.h"
#include "slam_datastructures/landmark.h"
#include "slam_datastructures/camera.h"
namespace cslam
{
void
SlamDebug::print_matches_from_lms(const Frame& frame1, const Frame& frame2,
                       const csfm::pyarray_uint8 image1, const csfm::pyarray_uint8 image2)
{
    const cv::Mat img1(image1.shape(0), image1.shape(1), CV_8UC3, (void *)image1.data());
    const cv::Mat img2(image2.shape(0), image2.shape(1), CV_8UC3, (void *)image2.data());

    //compute the matches
    std::vector<cv::KeyPoint> kp1;
    std::map<Landmark*, size_t> lm_to_kp1; // indices for matching
    std::vector<cv::DMatch> matches;
    for (size_t id1 = 0; id1 < frame1.landmarks_.size(); ++id1)
    {
        auto lm = frame1.landmarks_.at(id1);
        if (lm != nullptr)
        {
            lm_to_kp1[lm] = id1;
        }
    }

    for (size_t id2 = 0; id2 < frame2.landmarks_.size(); ++id2)
    {
        auto lm = frame2.landmarks_.at(id2);
        if (lm != nullptr)
        {
            auto it = lm_to_kp1.find(lm);
            if (it != lm_to_kp1.end())
                matches.emplace_back(it->second, id2, 0.0f);
        }
    }
    cv::Mat out;
    cv::drawMatches(img1, frame1.keypts_, img2,  frame2.keypts_, matches, out);
    cv::imshow("matches"+frame1.im_name+"<->"+frame2.im_name, out);
    cv::waitKey(0);

    // const cv::Mat img2 = (mask.shape(0) == 0 ? cv::Mat{} : cv::Mat(mask.shape(0), mask.shape(1), CV_8U, (void *)mask.data()));
}
// void
// SlamDebug::reproject_last_lms(const Frame& frame1, const Frame& frame2,
//                               const csfm::pyarray_uint8 image2, const BrownPerspectiveCamera& camera)
// {
//     const cv::Mat img2(image2.shape(0), image2.shape(1), CV_8UC3, (void *)image2.data());
//     for (size_t id1 = 0; id1 < frame1.landmarks_.size(); ++id1)
//     {
//         auto lm = frame1.landmarks_.at(id1);
//         if (!lm) {
//             continue;
//         }
//         // pose optimizationでoutlierになったものとは対応を取らない        
//         // Does not correspond to the outlier in pose optimization
//         if (last_frm.outlier_flags_.at(idx_last)) {
//             continue;
//         }
        

//         // グローバル基準の3次元点座標
//         // Global standard 3D point coordinates
//         const Eigen::Vector3f pos_w = lm->get_pos_in_world();

//         // 再投影して可視性を求める
//         // Reproject to find visibility
//         Eigen::Vector2f pt2D;


//         // float x_right;
//         // const bool in_image = camera_->reproject_to_image(rot_cw, trans_cw, pos_w, pt2D, x_right);
//         // const bool in_image = camera_.reproject_to_image(rot_cw, trans_cw, pos_w, pt2D);

//         // std::cout << "pos_w: " << pos_w << std::endl;
//         // 画像外に再投影される場合はスルー
//         // Thru if reprojected outside image
//         if (!camera.reproject_to_image(rot_cw, trans_cw, pos_w, grid_params_, pt2D))
//         { 
//             // std::cout << " out pt2D: " << pt2D.transpose() << std::endl;
//             continue;
//         }
//     }

// }

}