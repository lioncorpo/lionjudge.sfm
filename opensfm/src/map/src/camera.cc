#include <map/camera.h>
#include <map/shot.h>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

namespace map
{
// void
// Camera::UndistortedKeyptsToBearings(const std::vector<cv::KeyPoint> &undist_keypts,
//                                                     std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> bearings) const
// {

// }
// void
// Camera::UndistortKeypts(const std::vector<cv::KeyPoint> &keypts, std::vector<cv::KeyPoint> &undist_keypts) const
// {

// }

BrownPerspectiveCamera::BrownPerspectiveCamera(const size_t width_, const size_t height_, const std::string& projection_type_,
                                               const float fx_, const float fy_, const float cx_, const float cy_,
                                               const float k1_, const float k2_, const float p1_, const float p2_, const float k3_): 
                                               Camera(width_, height_, projection_type_), fx(fx_), fy(fy_), cx(cx_), cy(cy_),
                                               k1(k1_), k2(k2_), p1(p1_), p2(p2_), k3(k3_)
{
    K = (cv::Mat_<float>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    distCoeff = (cv::Mat_<float>(1,5) << k1, k2, p1, p2, k3);

    const auto s = std::max(width, height);
    cv::Mat norm_mat = (cv::Mat_<float>(3,3) << s, 0, 0.5*(width-1), 0, s, 0.5*(height-1), 0, 0, 1);
    
    K_pixel = norm_mat*K; //(cv::Mat_<float>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::cv2eigen(K_pixel, K_pixel_eig);
    fx_p = K_pixel_eig(0,0);
    fy_p = K_pixel_eig(1,1);
    cx_p = K_pixel_eig(0,2);
    cy_p = K_pixel_eig(1,2);

}
void
BrownPerspectiveCamera::UndistortedKeyptsToBearings(const AlignedVector<Observation>& undist_keypts, AlignedVector<Eigen::Vector3d>& bearings) const
{
  bearings.resize(undist_keypts.size());
  for (unsigned long idx = 0; idx < undist_keypts.size(); ++idx) {
      const auto x_normalized = (undist_keypts.at(idx).point[0] - cx_p) / fx_p;
      const auto y_normalized = (undist_keypts.at(idx).point[1] - cy_p) / fy_p;
      // const auto l2_norm = std::sqrt(x_normalized * x_normalized + y_normalized * y_normalized + 1.0);
      // bearings.at(idx) = Eigen::Vector3f{x_normalized / l2_norm, y_normalized / l2_norm, 1.0 / l2_norm};
      // std::cout << "oslam: " <<  bearings.at(idx) << " my: " << Eigen::Vector3f(x_normalized, y_normalized, 1.0).normalized() << std::endl;
      bearings.at(idx) = Eigen::Vector3d(x_normalized, y_normalized, 1.0).normalized();
  }
}
void
BrownPerspectiveCamera::UndistortKeypts(const AlignedVector<Observation>& keypts, AlignedVector<Observation>& undist_keypts) const
{
  const auto num_keypts = keypts.size();
  // Fill matrix with points
  cv::Mat upTmp(num_keypts,2,CV_32F);
  for(size_t i=0; i<num_keypts; i++)
  {
    upTmp.at<float>(i, 0) = keypts[i].point[0];
    upTmp.at<float>(i, 1) = keypts[i].point[1];
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
      undist_keypts.at(idx).point = Eigen::Vector2d(upTmp.at<float>(idx,0), upTmp.at<float>(idx,1));
    //   undist_keypts.at(idx).point.x = upTmp.at<float>(idx, 0);
    //   undist_keypts.at(idx).point.y = upTmp.at<float>(idx, 1);
      undist_keypts.at(idx).angle = keypts.at(idx).angle;
      undist_keypts.at(idx).size = keypts.at(idx).size;
      undist_keypts.at(idx).scale = keypts.at(idx).scale;
  }
}
bool 
BrownPerspectiveCamera::ReprojectToImage(const Eigen::Matrix3d& R_cw, const Eigen::Vector3d& t_cw, const Eigen::Vector3d& ptWorld,
                                         Eigen::Vector2d& pt2D) const 
{
    const Eigen::Vector3d ptCam = R_cw*ptWorld + t_cw;
    //check z coordinate
    if (ptCam[2] < 0.0) return false;
    // //now reproject to image
    pt2D = (K_pixel_eig.cast<double>()*ptCam).hnormalized();
    return true;
};

bool 
BrownPerspectiveCamera::ReprojectToImage(const Eigen::Matrix3f& R_cw, const Eigen::Vector3f& t_cw, const Eigen::Vector3f& ptWorld,
                                         Eigen::Vector2f& pt2D) const
{
    //first, transform pt3D into cam
    const Eigen::Vector3f ptCam = R_cw*ptWorld + t_cw;
    //check z coordinate
    if (ptCam[2] < 0.0) return false;
    // //now reproject to image
    pt2D = (K_pixel_eig*ptCam).hnormalized();
    return true;
    // return (K_pixel_eig*ptCam).hnormalized();
    // bearing.normalized();
    //check boundaries
    // TODO: think about different cameras and what the boundaries are actually! 
    // return (pt2D[0] >= 0 && pt2D[1] >= 0 && pt2D[0] < width && pt2D[1] < height);
    // return gridParams.in_grid(pt2D);
}
bool 
BrownPerspectiveCamera::ReprojectToBearing(const Eigen::Matrix3f& R_cw, const Eigen::Vector3f& t_cw, const Eigen::Vector3f& ptWorld,
                                            Eigen::Vector3f& bearing, Eigen::Vector2f& pt2D) const
{
    //first, transform pt3D into cam
    bearing = R_cw*ptWorld + t_cw;
    //check z coordinate
    if (bearing[2] < 0.0) return false;
    // //now reproject to image
    pt2D = (K_pixel_eig*bearing).hnormalized();
    bearing.normalized();
    return true;
}

bool 
BrownPerspectiveCamera::ReprojectToBearing(const Eigen::Matrix3d& R_cw, const Eigen::Vector3d& t_cw, const Eigen::Vector3d& ptWorld,
                                            Eigen::Vector3d& bearing, Eigen::Vector2d& pt2D) const
{
    //first, transform pt3D into cam
    bearing = R_cw*ptWorld + t_cw;
    //check z coordinate
    if (bearing[2] < 0.0) return false;
    // //now reproject to image
    pt2D = (K_pixel_eig.cast<double>()*bearing).hnormalized();
    bearing.normalized();
    return true;
}


} // namespace map