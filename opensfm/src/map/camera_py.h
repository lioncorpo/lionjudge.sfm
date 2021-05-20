#pragma once

#include <map/camera.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
namespace map {

class PyCamera : public Camera 
{
public:
    using Camera::Camera;
    Eigen::Vector3d PixelBearing(const Eigen::Vector2d& point) const override
    {
        PYBIND11_OVERLOAD_PURE(Eigen::Vector3d, Camera, PixelBearing, point);
    }
};

// class PyBrownPerspectiveCamera : public BrownPerspectiveCamera 
// {
//  public:
//     using BrownPerspectiveCamera::BrownPerspectiveCamera;
//     Eigen::Vector3d PixelBearing(const Eigen::Vector2d& point) const override
//     {
//         PYBIND11_OVERLOAD(Eigen::Vector3d, BrownPerspectiveCamera, PixelBearing, point );
//     }   
// };

// class PyPerspectiveCamera : public PerspectiveCamera 
// {
// public:
//     using PerspectiveCamera::PerspectiveCamera;
//     Eigen::Vector3d PixelBearing(const Eigen::Vector2d& point) const override
//     {
//         PYBIND11_OVERLOAD(Eigen::Vector3d, PerspectiveCamera, PixelBearing, point );
//     }
// };
}  // namespace map