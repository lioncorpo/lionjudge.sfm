#include <geometry/camera.h>
#include <geometry/camera_functions.h>
#include <iostream>

Camera Camera::CreatePerspectiveCamera(double focal, double k1, double k2) {
  VecXd parameters(3);
  parameters << focal, k1, k2;
  return Camera(ProjectionType::PERSPECTIVE, parameters);
};

Camera Camera::CreateBrownCamera(double focal, double aspect_ratio,
                                 const Eigen::Vector2d& principal_point,
                                 const Eigen::VectorXd& distortion) {
  if (distortion.size() != 5) {
    throw std::runtime_error("Invalid distortion coefficients size");
  }
  VecXd parameters(9);
  parameters << focal, aspect_ratio, principal_point[0],
      principal_point[1], distortion[0], distortion[1], distortion[2],
      distortion[3], distortion[4];
  return Camera(ProjectionType::BROWN, parameters);
};

Camera Camera::CreateFisheyeCamera(double focal, double k1, double k2) {
  VecXd parameters(3);
  parameters << focal, k1, k2;
  return Camera(ProjectionType::FISHEYE, parameters);
};

Camera Camera::CreateDualCamera(double transition, double focal, double k1,
                                double k2) {
  VecXd parameters(4);
  parameters << transition, focal, k1, k2;
  return Camera(ProjectionType::DUAL, parameters);
};

Camera Camera::CreateSphericalCamera() {
  VecXd parameters(0);
  return Camera(ProjectionType::SPHERICAL, parameters);
};

ProjectionType Camera::GetProjectionType() const { return type_; }

std::string Camera::GetProjectionString() const {
  switch (type_) {
    case ProjectionType::PERSPECTIVE:
      return "perspective";
    case ProjectionType::BROWN:
      return "brown";
    case ProjectionType::FISHEYE:
      return "fisheye";
    case ProjectionType::DUAL:
      return "dual";
    case ProjectionType::SPHERICAL:
      return "spherical";
    default:
      throw std::runtime_error("Invalid ProjectionType");
  }
}

VecXd Camera::GetParameters() const {
  return parameters_;
}

void Camera::SetParameters(const VecXd &p) {
  parameters_ = p;
}

Eigen::Matrix3d Camera::GetProjectionMatrix() const {
  switch (type_) {
    case ProjectionType::PERSPECTIVE: {
      Mat3d K;
      double f = parameters_[0];
      K << f, 0, 0, 0, f, 0, 0, 0, 1;
      return K;
    }
    case ProjectionType::BROWN: {
      Mat3d K;
      double f = parameters_[0];
      double a = parameters_[1];
      double px = parameters_[2];
      double py = parameters_[3];
      K << f, 0, px, 0, a * f, py, 0, 0, 1;
      return K;
    }
    default:
      throw std::runtime_error("Invalid ProjectionType");
  }
}

Eigen::Matrix3d Camera::GetProjectionMatrixScaled(int width, int height) const {
  const int size = std::max(width, height);

  Eigen::Matrix3d K = GetProjectionMatrix();
  Mat3d H;
  H << size, 0, (width - 1) / 2.0,
       0, size, (height - 1) / 2.0,
       0, 0, 1;
  return H * K;
}

Eigen::Vector2d Camera::Project(const Eigen::Vector3d& point) const {
  return Dispatch<Eigen::Vector2d, ProjectFunction>(
      type_, point, parameters_);
}

Eigen::MatrixX2d Camera::ProjectMany(const Eigen::MatrixX3d& points) const {
  Eigen::MatrixX2d projected(points.rows(), 2);
  for (int i = 0; i < points.rows(); ++i) {
    projected.row(i) = Project(points.row(i));
  }
  return projected;
}

Eigen::Vector3d Camera::Bearing(const Eigen::Vector2d& point) const {
  return Dispatch<Eigen::Vector3d, BearingFunction>(
      type_, point, parameters_);
}

Eigen::MatrixX3d Camera::BearingsMany(const Eigen::MatrixX2d& points) const {
  Eigen::MatrixX3d projected(points.rows(), 3);
  for (int i = 0; i < points.rows(); ++i) {
    projected.row(i) = Bearing(points.row(i));
  }
  return projected;
}

Camera::Camera(ProjectionType type, VecXd parameters)
    : type_(type), parameters_(parameters) {}

std::pair<Eigen::MatrixXf, Eigen::MatrixXf> ComputeCameraMapping(const Camera& from, const Camera& to, int width, int height){
  const auto normalizer_factor = std::max(width, height);
  const auto inv_normalizer_factor = 1.0/normalizer_factor;

  Eigen::MatrixXf u_from(height, width);
  Eigen::MatrixXf v_from(height, width);

  const auto half_width = width*0.5;
  const auto half_height = height*0.5;

  for(int v = 0; v < height; ++v){
    for(int u = 0; u < width; ++u){
      const auto uv = Eigen::Vector2d(u-half_width, v-half_height);
      const Eigen::Vector2d point_uv_from = normalizer_factor*from.Project(to.Bearing(inv_normalizer_factor*uv));
      u_from(v, u) = point_uv_from(0) + half_width;
      v_from(v, u) = point_uv_from(1) + half_height;
    }
  }
  return std::make_pair(u_from, v_from);
}
