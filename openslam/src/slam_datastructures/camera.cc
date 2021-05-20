#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include "camera.h"
#include "frame.h"
#include "third_party/openvslam/util/guided_matching.h"
namespace cslam
{


void
BrownPerspectiveCamera::undistKeyptsFrame(Frame& frame) const
{
    undistKeypts(frame.keypts_, frame.undist_keypts_);
}
void
BrownPerspectiveCamera::convertKeyptsToBearingsFrame(Frame& frame) const
{
    convertKeyptsToBearings(frame.undist_keypts_, frame.bearings_);
}
void
BrownPerspectiveCamera::convertKeyptsToBearings(const std::vector<cv::KeyPoint>& undist_keypts,
std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& bearings) const
{
    bearings.resize(undist_keypts.size());
    for (unsigned long idx = 0; idx < undist_keypts.size(); ++idx) {
        const auto x_normalized = (undist_keypts.at(idx).pt.x - cx_p) / fx_p;
        const auto y_normalized = (undist_keypts.at(idx).pt.y - cy_p) / fy_p;
        // const auto l2_norm = std::sqrt(x_normalized * x_normalized + y_normalized * y_normalized + 1.0);
        // bearings.at(idx) = Eigen::Vector3f{x_normalized / l2_norm, y_normalized / l2_norm, 1.0 / l2_norm};
        // std::cout << "oslam: " <<  bearings.at(idx) << " my: " << Eigen::Vector3f(x_normalized, y_normalized, 1.0).normalized() << std::endl;
        bearings.at(idx) = Eigen::Vector3f(x_normalized, y_normalized, 1.0).normalized();
    }
}

void 
BrownPerspectiveCamera::undistKeypts(const std::vector<cv::KeyPoint>& keypts, std::vector<cv::KeyPoint>& undist_keypts) const
{
    //TODO: 0 distortion?

    const auto num_keypts = keypts.size();
    // Fill matrix with points
    cv::Mat upTmp(num_keypts,2,CV_32F);
    for(size_t i=0; i<num_keypts; i++)
    {
        upTmp.at<float>(i,0)=keypts[i].pt.x;
        upTmp.at<float>(i,1)=keypts[i].pt.y;
    }
    // Undistort points
    upTmp=upTmp.reshape(2);
    cv::undistortPoints(upTmp,upTmp,K,distCoeff,cv::Mat(),K);
    upTmp=upTmp.reshape(1);
    // std::cout << "undist_keypts: " << undist_keypts.size() << std::endl;
    undist_keypts.resize(num_keypts);
    // std::cout << "num_keypts: " << num_keypts << " keypts: " << keypts.size() << std::endl;
    for(size_t idx = 0; idx < num_keypts; idx++)
    {
        undist_keypts.at(idx).pt.x = upTmp.at<float>(idx, 0);
        undist_keypts.at(idx).pt.y = upTmp.at<float>(idx, 1);
        undist_keypts.at(idx).angle = keypts.at(idx).angle;
        undist_keypts.at(idx).size = keypts.at(idx).size;
        undist_keypts.at(idx).octave = keypts.at(idx).octave;
    }
    // std::cout << "undist_keypts: " << undist_keypts.size() << std::endl;
}
// bool 
// BrownPerspectiveCamera::reproject_to_image_dist(const Eigen::Matrix3f& R_cw, const Eigen::Vector3f& t_cw, const Eigen::Vector3f& ptWorld,
//                                          const cslam::GridParameters& gridParams, Eigen::Vector2f& pt2D) const
// {
//     cv::projectPoints()
// }
bool 
BrownPerspectiveCamera::reproject_to_image(const Eigen::Matrix3f& R_cw, const Eigen::Vector3f& t_cw, const Eigen::Vector3f& ptWorld,
                                            const cslam::GridParameters& gridParams,
                                            Eigen::Vector2f& pt2D) const
{
    //first, transform pt3D into cam
    const Eigen::Vector3f ptCam = R_cw*ptWorld + t_cw;
    //check z coordinate
    if (ptCam[2] < 0.0) return false;
    // //now reproject to image
    pt2D = (K_pixel_eig*ptCam).hnormalized();
    // bearing.normalized();
    //check boundaries
    return gridParams.in_grid(pt2D);
}
bool 
BrownPerspectiveCamera::reproject_to_bearing(const Eigen::Matrix3f& R_cw, const Eigen::Vector3f& t_cw, const Eigen::Vector3f& ptWorld,
                                            const cslam::GridParameters& gridParams,
                                            Eigen::Vector3f& bearing) const
{
    //first, transform pt3D into cam
    bearing = R_cw*ptWorld + t_cw;
    //check z coordinate
    if (bearing[2] < 0.0) return false;
    // //now reproject to image
    const Eigen::Vector2f pt2D = (K_pixel_eig*bearing).hnormalized();
    bearing.normalized();
    //check boundaries
    return gridParams.in_grid(pt2D);
}
}