#pragma once
#include <Eigen/Core>

namespace map
{
// TODO: unify with sfm Observation
struct Observation
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Observation() = default;
  Observation(double x, double y, double s, uint8_t r, uint8_t g, uint8_t b, size_t id)
      : point(x, y), scale(s), color(r, g, b), id(id),
        angle(0), response(0), size(0), class_id(-1)
        { }
  Observation(double x, double y, double s, uint8_t r, uint8_t g, uint8_t b, size_t id,
              float _angle, float _resp, float _size, int _class_id)
    : point(x, y), scale(s), color(r, g, b), id(id),
      angle(_angle), response(_resp), size(_size), class_id(_class_id)
      { }
  bool operator==(const Observation &k) const
  {
    return point == k.point && scale == k.scale && color == k.color &&
           id == k.id;
  }
  Eigen::Vector2d point;
  double scale{1}; //same as octave
  Eigen::Matrix<uint8_t, 3, 1> color;
  size_t id{0};


  float angle;
  float response;
  // int octave; //this is probably the scale
  float size;
  int class_id;
};
} // namespace map