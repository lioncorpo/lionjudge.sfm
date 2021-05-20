#pragma once

#include <initializer_list>

#include "ceres/ceres.h"
#include "ceres/rotation.h"

template <class T>
void MultRotations(const T R1[3], const T R2[3], T result[3]) {
  T qR1[4], qR2[4], qResult[4];
  ceres::AngleAxisToQuaternion(R1, qR1);
  ceres::AngleAxisToQuaternion(R2, qR2);
  ceres::QuaternionProduct(qR1, qR2, qResult);
  ceres::QuaternionToAngleAxis(qResult, result);
}

template <class T>
void MultRotations(const T R1[3], const T R2[3], const T R3[3], T result[3]) {
  T qR1[4], qR2[4], qR3[4], qR1R2[4], qResult[4];
  ceres::AngleAxisToQuaternion(R1, qR1);
  ceres::AngleAxisToQuaternion(R2, qR2);
  ceres::AngleAxisToQuaternion(R3, qR3);
  ceres::QuaternionProduct(qR1, qR2, qR1R2);
  ceres::QuaternionProduct(qR1R2, qR3, qResult);
  ceres::QuaternionToAngleAxis(qResult, result);
}

/* apply a rotation R to a vector x as R*x rotations is expected to be
 * angle-axis */
template <typename T>
Vec3<T> RotatePoint(const Vec3<T>& R, const Vec3<T>& x) {
  Vec3<T> rotated;
  ceres::AngleAxisRotatePoint(R.data(), x.data(), rotated.data());
  return rotated;
}

/* bring a point x in the coordinate frame of a camera with rotation and camera
 * center in world coordinates being respectively R and c such : x(camera) =
 * R(t).(x(world) - c) */
template <typename T>
Vec3<T> WorldToCamera(const Vec3<T>& R, const Vec3<T>& c, const Vec3<T>& x) {
  return RotatePoint((-R).eval(), (x - c).eval());
}

/* apply a similarity transform of scale s, rotation R and translation t to some
 * point x as s * R * x + t */
template <class T>
Vec3<T> ApplySimilarity(const T& s, const Vec3<T>& R, const Vec3<T>& t,
                        const Vec3<T>& x) {
  return RotatePoint((-R).eval(), (s * x - t).eval());
}