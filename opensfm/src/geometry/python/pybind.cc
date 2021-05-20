#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <glog/logging.h>

#include <geometry/essential.h>
#include <geometry/camera.h>
#include <geometry/relative_pose.h>
#include <geometry/absolute_pose.h>
#include <geometry/triangulation.h>
#include <foundation/python_types.h>


PYBIND11_MODULE(pygeometry, m) {

  py::enum_<ProjectionType>(m, "ProjectionType")
    .value("PERSPECTIVE", ProjectionType::PERSPECTIVE)
    .value("BROWN", ProjectionType::BROWN)
    .value("FISHEYE", ProjectionType::FISHEYE)
    .value("DUAL", ProjectionType::DUAL)
    .value("SPHERICAL", ProjectionType::SPHERICAL)
    .export_values()
  ;

  py::class_<Camera>(m, "Camera")
  .def_static("create_perspective", &Camera::CreatePerspectiveCamera)
  .def_static("create_brown", &Camera::CreateBrownCamera)
  .def_static("create_fisheye", &Camera::CreateFisheyeCamera)
  .def_static("create_dual", &Camera::CreateDualCamera)
  .def_static("create_spherical", &Camera::CreateSphericalCamera)
  .def("project", &Camera::Project)
  .def("project_many", &Camera::ProjectMany)
  .def("pixel_bearing", &Camera::Bearing)
  .def("pixel_bearing_many", &Camera::BearingsMany)
  .def("get_K", &Camera::GetProjectionMatrix)
  .def("get_K_in_pixel_coordinates", &Camera::GetProjectionMatrixScaled)
  .def_readwrite("width", &Camera::width)
  .def_readwrite("height", &Camera::height)
  .def_readwrite("id", &Camera::id)
  .def_property("parameters", &Camera::GetParameters, &Camera::SetParameters)
  .def_property_readonly("projection_type", &Camera::GetProjectionString)
  .def(py::pickle(
    [](const Camera &p) {
      return py::make_tuple(p.GetParameters(), p.GetProjectionType(),
                            p.width, p.height, p.id);
    },
    [](py::tuple t) {
      const auto parameters = t[0].cast<Eigen::VectorXd>();
      const auto type = t[1].cast<ProjectionType>();
      const auto width = t[2].cast<int>();
      const auto height = t[3].cast<int>();
      const auto id = t[4].cast<std::string>();
      Camera camera(type, parameters);
      camera.width = width;
      camera.height = height;
      camera.id = id;
      return camera;
    }))
  // Python2 + copy/deepcopy + pybind11 workaround
  .def("__copy__", [](const Camera& c, const py::dict& d){ return c;}, py::return_value_policy::copy)
  .def("__deepcopy__", [](const Camera& c, const py::dict& d){ return c;}, py::return_value_policy::copy)
  ;
  m.def("compute_camera_mapping", ComputeCameraMapping);

  m.def("triangulate_bearings_dlt", geometry::TriangulateBearingsDLT);
  m.def("triangulate_bearings_midpoint", geometry::TriangulateBearingsMidpoint);
  m.def("triangulate_two_bearings_midpoint", geometry::TriangulateTwoBearingsMidpointSolve<double>);
  m.def("triangulate_two_bearings_midpoint_many", geometry::TriangulateTwoBearingsMidpointMany);
  m.def("essential_five_points", geometry::EssentialFivePoints);
  m.def("absolute_pose_three_points", geometry::AbsolutePoseThreePoints);
  m.def("absolute_pose_n_points", geometry::AbsolutePoseNPoints);
  m.def("absolute_pose_n_points_known_rotation", geometry::AbsolutePoseNPointsKnownRotation);
  m.def("essential_n_points", geometry::EssentialNPoints);
  m.def("relative_pose_from_essential", geometry::RelativePoseFromEssential);
  m.def("relative_rotation_n_points", geometry::RelativeRotationNPoints);
  m.def("relative_pose_refinement", geometry::RelativePoseRefinement);
}
