#pragma once

#include <foundation/types.h>
#include <geometry/camera_functions.h>
#include <Eigen/Eigen>

class Camera {
 public:
  Camera(ProjectionType t, VecXd parameters);
  static Camera CreatePerspectiveCamera(double focal, double k1, double k2);
  static Camera CreateBrownCamera(double focal, double aspect_ratio,
                                  const Eigen::Vector2d& principal_point,
                                  const Eigen::VectorXd& distortion);
  static Camera CreateFisheyeCamera(double focal, double k1, double k2);
  static Camera CreateDualCamera(double transition, double focal, double k1, double k2);
  static Camera CreateSphericalCamera();

  Eigen::Vector2d Project(const Eigen::Vector3d& point) const;
  Eigen::MatrixX2d ProjectMany(const Eigen::MatrixX3d& points) const;

  Eigen::Vector3d Bearing(const Eigen::Vector2d& point) const;
  Eigen::MatrixX3d BearingsMany(const Eigen::MatrixX2d& points) const;

  ProjectionType GetProjectionType()const;
  std::string GetProjectionString()const;

  VecXd GetParameters() const;
  void SetParameters(const VecXd &p);

  Eigen::Matrix3d GetProjectionMatrix()const;
  Eigen::Matrix3d GetProjectionMatrixScaled(int width, int height)const;
  
  int width{1};
  int height{1};
  std::string id;

 private:
  ProjectionType type_;
  VecXd parameters_;
};

std::pair<Eigen::MatrixXf, Eigen::MatrixXf> ComputeCameraMapping(const Camera& from, const Camera& to, int width, int height);
