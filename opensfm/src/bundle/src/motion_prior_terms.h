#pragma once

#include "error_utils.h"

#include <Eigen/Eigen>

struct BALinearMotionError {
  BALinearMotionError(double alpha,
                      double position_std_deviation,
                      double orientation_std_deviation)
      : alpha_(alpha)
      , position_scale_(1.0 / position_std_deviation)
      , orientation_scale_(1.0 / orientation_std_deviation)
  {}

  template <typename T>
  bool operator()(const T* const shot0,
                  const T* const shot1,
                  const T* const shot2,
                  T* r) const {
    const T* R0 = shot0 + BA_SHOT_RX;
    Eigen::Map<const Vec3<T> > t0(shot0 + BA_SHOT_TX);
    const T* R1 = shot1 + BA_SHOT_RX;
    Eigen::Map< const Vec3<T> > t1(shot1 + BA_SHOT_TX);
    const T* R2 = shot2 + BA_SHOT_RX;
    Eigen::Map< const Vec3<T> > t2(shot2 + BA_SHOT_TX);

    // Residual have the general form :
    //  op( alpha . op(2, -0), op(0, -1))
    // with - being the opposite
    Eigen::Map< Eigen::Matrix<T,6,1> > residual(r);

    // Position residual : op is translation
    residual.segment(0, 3) = T(position_scale_) * (T(alpha_) * (t2 - t0) + (t0 - t1));

    // Rotation residual : op is rotation
    const T R0t[3] = {-R0[0], -R0[1], -R0[2]};
    const T R1t[3] = {-R1[0], -R1[1], -R1[2]};
    T R2_R0t[3], R0_R1t[3], dR[3];
    MultRotations(R2, R0t, R2_R0t);
    MultRotations(R0, R1t, R0_R1t);
    T alpha_R2_R0t[3] = {
        T(alpha_) * R2_R0t[0],
        T(alpha_) * R2_R0t[1],
        T(alpha_) * R2_R0t[2],
    };
    MultRotations(alpha_R2_R0t, R0_R1t, dR);

    residual[3] = T(position_scale_) * dR[0];
    residual[4] = T(position_scale_) * dR[1];
    residual[5] = T(position_scale_) * dR[2];
    return true;
  }

  Vec3d acceleration_;
  double position_scale_;
  double orientation_scale_;
  double alpha_;
};