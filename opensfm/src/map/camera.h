#pragma once
#include <map/defines.h>
#include <map/observation.h>
#include <vector>
#include <opencv2/core.hpp>
#include <Eigen/Core>
namespace map
{
class Camera
{
public:
  Camera(const size_t width_, const size_t height_, const std::string& projection_type_):
          width(width_), height(height_), projectionType(projection_type_)
  {}
  virtual ~Camera() = default;

  const size_t width;
  const size_t height;
  //TODO: Make to enum
  const std::string projectionType;
  // virtual void UndistortedKeyptsToBearings(const std::vector<cv::KeyPoint>& undistKeypts,
  //                                     std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& bearings) const {};
  // virtual void UndistortKeypts(const std::vector<cv::KeyPoint>& keypts, std::vector<cv::KeyPoint>& undist_keypts) const {};
  virtual void UndistortedKeyptsToBearings(const AlignedVector<Observation>& undistKeypts, AlignedVector<Eigen::Vector3d>& bearings) const {};
  virtual void UndistortKeypts(const AlignedVector<Observation>& keypts, AlignedVector<Observation>& undist_keypts) const {};
  virtual bool ReprojectToImage(const Eigen::Matrix3f& R_cw, const Eigen::Vector3f& t_cw, const Eigen::Vector3f& ptWorld,
                                Eigen::Vector2f& pt2D) const {return true;};
  virtual bool ReprojectToImage(const Eigen::Matrix3d& R_cw, const Eigen::Vector3d& t_cw, const Eigen::Vector3d& ptWorld,
                                Eigen::Vector2d& pt2D) const {return true;};
  virtual bool ReprojectToBearing(const Eigen::Matrix3f& R_cw, const Eigen::Vector3f& t_cw, const Eigen::Vector3f& ptWorld,
                                  Eigen::Vector3f& bearing, Eigen::Vector2f& pt2D) const { return true; }
  virtual bool ReprojectToBearing(const Eigen::Matrix3d& R_cw, const Eigen::Vector3d& t_cw, const Eigen::Vector3d& ptWorld,
                                Eigen::Vector3d& bearing, Eigen::Vector2d& pt2D) const { return true; }
  Eigen::Vector3d NormalizePointAndScale(const Eigen::Vector3d& point_and_scale) const
  {
    return NormalizePointAndScale(point_and_scale.head<2>(), point_and_scale[2]);
  }

  Eigen::Vector3d NormalizePointAndScale(const Eigen::Vector2d& point, const double scale) const
  {
    const auto size = std::max(width, height);
    return Eigen::Vector3d(point[0] + 0.5 - width/2.0,
                           point[1] + 0.5 - height/2.0,
                           scale) / size;
  }
  Eigen::Vector2d NormalizePoint(const Eigen::Vector2d& point) const
  {
    const auto size = std::max(width, height);
    return Eigen::Vector2d(point[0] + 0.5 - width/2.0,
                           point[1] + 0.5 - height/2.0) / size;
  }
  Eigen::Vector2d DeNormalizePoint(const Eigen::Vector2d& point) const
  {
    const auto size = std::max(width, height);
    return Eigen::Vector2d(point[0]*size - 0.5 + width/2.0,
                           point[1]*size - 0.5 + height/2.0);
  }

  bool InImage(const Eigen::Vector2d& point) const
  {
    return point[0] >= 0 && point[1] >= 0 && point[0] < width && point[1] < height;
  }
};

class BrownPerspectiveCamera : public Camera
{
public:
  BrownPerspectiveCamera(const size_t width_, const size_t height_, const std::string& projection_type_,
                         const float fx_, const float fy_, const float cx_, const float cy_,
                         const float k1_, const float k2_, const float p1_, const float p2_, const float k3_);
  virtual void UndistortedKeyptsToBearings(const AlignedVector<Observation>& undist_keypts, AlignedVector<Eigen::Vector3d>& bearings) const;
  virtual void UndistortKeypts(const AlignedVector<Observation>& keypts, AlignedVector<Observation>& undist_keypts) const;
  virtual bool ReprojectToImage(const Eigen::Matrix3f& R_cw, const Eigen::Vector3f& t_cw, const Eigen::Vector3f& ptWorld,
                                Eigen::Vector2f& pt2D) const;
  virtual bool ReprojectToImage(const Eigen::Matrix3d& R_cw, const Eigen::Vector3d& t_cw, const Eigen::Vector3d& ptWorld,
                                Eigen::Vector2d& pt2D) const;
  
  virtual bool ReprojectToBearing(const Eigen::Matrix3f& R_cw, const Eigen::Vector3f& t_cw, const Eigen::Vector3f& ptWorld,
                                  Eigen::Vector3f& bearing, Eigen::Vector2f& pt2D) const;
  virtual bool ReprojectToBearing(const Eigen::Matrix3d& R_cw, const Eigen::Vector3d& t_cw, const Eigen::Vector3d& ptWorld,
                                  Eigen::Vector3d& bearing, Eigen::Vector2d& pt2D) const;

  float fx_p, fy_p; // focal lengths in pixels
  float cx_p, cy_p; // principal points in pixels
  
  const float fx, fy; // focal lengths
  const float cx, cy; // principal points
  const float k1, k2, p1, p2, k3; // distortion coefficients
  cv::Mat K, K_pixel; //intrinsic camera matrix
  cv::Mat distCoeff; //distortion coefficients
  Eigen::Matrix3f K_pixel_eig;
};

}; //end reconstruction