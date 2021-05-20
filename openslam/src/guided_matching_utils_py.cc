#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "third_party/openvslam/util/guided_matching.h"
#include "slam_datastructures/frame.h"
#include "slam_datastructures/keyframe.h"
#include "slam_datastructures/landmark.h"
namespace py = pybind11;


PYBIND11_MODULE(guided_matching, m) {
    py::class_<guided_matching::GridParameters>(m, "GridParameters")
    .def(py::init<unsigned int, unsigned int, float, float, float, float>());

    m.def("assign_keypoints_to_grid", &guided_matching::assign_keypoints_to_grid);
    m.def("distribute_keypoints_to_grid_frame", &guided_matching::distribute_keypoints_to_grid_frame);
    m.def("match_frame_to_frame_py", &guided_matching::match_frame_to_frame_py);
    // m.def("match_frame_to_frame_dbg", &guided_matching::match_frame_to_frame_dbg);
    m.def("match_frame_to_frame", &guided_matching::match_frame_to_frame);
    m.def("match_frame_and_landmarks", 
        &guided_matching::match_frame_and_landmarks);
}
