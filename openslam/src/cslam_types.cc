// #pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "third_party/openvslam/feature/orb_extractor.h"
#include "slam_datastructures/frame.h"
#include "slam_datastructures/keyframe.h"
#include "slam_datastructures/landmark.h"
#include "slam_datastructures/camera.h"
#include "types.h"

namespace py = pybind11;
// Define the bindings!
PYBIND11_MODULE(cslam_types, m) {
        // Frame(csfm::pyarray_uint8 image, csfm::pyarray_uint8 mask, const std::string& img_name,
        //   const size_t frame_id, const openvslam::feature::orb_extractor& orb_extractor);
    // Frame
    py::class_<cslam::Frame>(m, "Frame")
        // .def(py::init< const std::string&, const size_t, const std::vector<float>>())
        .def(py::init<const csfm::pyarray_uint8, const csfm::pyarray_uint8, const std::string&,
                      const size_t, openvslam::feature::orb_extractor*>())
        .def_readonly("frame_id", &cslam::Frame::frame_id)
        .def_readonly("im_name", &cslam::Frame::im_name)
        // .def_readonly("keypts", &cslam::Frame::keypts_)
        .def("getKptsAndDescPy",&cslam::Frame::getKptsAndDescPy)
        .def("getKptsPy",&cslam::Frame::getKptsPy)
        .def("getDescPy",&cslam::Frame::getDescPy)
        .def("getKptsUndist", &cslam::Frame::getKptsUndist);
    // Keyframe
    py::class_<cslam::KeyFrame>(m, "KeyFrame")
        .def(py::init<const size_t, const cslam::Frame>())
        .def_readonly("kf_id", &cslam::KeyFrame::kf_id_)
        .def("set_pose", &cslam::KeyFrame::set_Twc)
        .def("get_pose", &cslam::KeyFrame::get_Twc);

    // Landmark
    py::class_<cslam::Landmark>(m, "Landmark")
        .def(py::init<const size_t, cslam::KeyFrame*, const Eigen::Vector3f&>())
        .def("increase_num_observable", &cslam::Landmark::increase_num_observable)
        .def("increase_num_observed", &cslam::Landmark::increase_num_observed)
        .def("get_observed_ratio", &cslam::Landmark::get_observed_ratio)
        .def("add_observation", &cslam::Landmark::add_observation)
        .def("update_normal_and_depth", &cslam::Landmark::update_normal_and_depth)
        .def("compute_descriptor", &cslam::Landmark::compute_descriptor);


    //Camera
    py::class_<cslam::BrownPerspectiveCamera>(m, "BrownPerspectiveCamera")
        .def(py::init<const size_t, const size_t, const std::string&,
                      const float, const float, const float, const float,
                      const float, const float, const float, const float, const float>())
        .def("undistKeyptsFrame", &cslam::BrownPerspectiveCamera::undistKeyptsFrame);


        // (const size_t width_, const size_t height_, const std::string& projection_type_,
        //  const float fx_, const float fy_, const float cx_, const float cy_,
        //  const float k1_, const float k2_, const float p1_, const float p2_, const float k3_)
}