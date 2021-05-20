#include <glog/logging.h>
#include <map/camera.h>
#include <map/camera_py.h>
#include <map/defines.h>
#include <map/landmark.h>
#include <map/map.h>
#include <map/map_io.h>
#include <map/observation.h>
#include <map/pose.h>
#include <map/shot.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
PYBIND11_MODULE(pymap, m) {
  py::class_<map::Pose>(m, "Pose")
      .def(py::init())
      .def("get_cam_to_world", &map::Pose::CameraToWorld)
      .def("get_world_to_cam", &map::Pose::WorldToCamera)
      .def("set_from_cam_to_world", py::overload_cast<const Eigen::Matrix4d &>(
                                        &map::Pose::SetFromCameraToWorld))
      .def("set_from_cam_to_world",
           py::overload_cast<const Eigen::Matrix3d &, const Eigen::Vector3d &>(
               &map::Pose::SetFromCameraToWorld))
      .def("set_from_cam_to_world",
           py::overload_cast<const Eigen::Vector3d &, const Eigen::Vector3d &>(
               &map::Pose::SetFromCameraToWorld))
      .def("set_from_world_to_cam", py::overload_cast<const Eigen::Matrix4d &>(
                                        &map::Pose::SetFromWorldToCamera))
      .def("set_from_world_to_cam",
           py::overload_cast<const Eigen::Matrix3d &, const Eigen::Vector3d &>(
               &map::Pose::SetFromWorldToCamera))
      .def("set_from_world_to_cam",
           py::overload_cast<const Eigen::Vector3d &, const Eigen::Vector3d &>(
               &map::Pose::SetFromWorldToCamera))
      .def("get_origin", &map::Pose::GetOrigin)
      .def("get_R_cam_to_world", &map::Pose::RotationCameraToWorld)
      .def("get_rotation_matrix", &map::Pose::RotationWorldToCamera)
      .def("get_R_world_to_cam", &map::Pose::RotationWorldToCamera)
      .def("get_R_cam_to_world_min", &map::Pose::RotationCameraToWorldMin)
      .def("get_R_world_to_cam_min", &map::Pose::RotationWorldToCameraMin)
      .def("get_t_cam_to_world", &map::Pose::TranslationCameraToWorld)
      .def("get_t_world_to_cam", &map::Pose::TranslationWorldToCamera)
      .def_property("rotation", &map::Pose::RotationWorldToCameraMin,
                    &map::Pose::SetWorldToCamRotation)
      .def_property("translation", &map::Pose::TranslationWorldToCamera,
                    &map::Pose::SetWorldToCamTranslation)
      .def("set_rotation_matrix", &map::Pose::SetWorldToCamRotationMatrix)
      .def("transform", &map::Pose::TransformWorldToCamera)
      .def("transform_inverse", &map::Pose::TransformCameraToWorld)
      ;

  py::class_<map::MapIO>(m, "MapIO")
      .def("save_map", &map::MapIO::SaveMapToFile)
      .def("color_map", &map::MapIO::ColorMap);
  py::class_<map::Map>(m, "Map")
      .def(py::init())
      .def("number_of_shots", &map::Map::NumberOfShots,
           "Returns the number of shots")
      .def("number_of_landmarks", &map::Map::NumberOfLandmarks)
      .def("number_of_cameras", &map::Map::NumberOfCameras)
      .def("create_shot_camera", &map::Map::CreateShotCamera, py::arg("cam_id"),
           py::arg("camera"), py::arg("name") = "",
           py::return_value_policy::reference_internal)
      .def("remove_shot_camera", &map::Map::RemoveShotCamera)
      // Landmark
      .def("create_landmark", &map::Map::CreateLandmark, py::arg("lm_id"),
           py::arg("global_position"),
           py::return_value_policy::reference_internal)
      .def("update_landmark", &map::Map::UpdateLandmark)
      .def("remove_landmark", py::overload_cast<const map::Landmark *const>(
                                  &map::Map::RemoveLandmark))
      .def("remove_landmark",
           py::overload_cast<const map::LandmarkId>(&map::Map::RemoveLandmark))
      // Shot
      .def("create_shot",
           py::overload_cast<const map::ShotId, const map::CameraId,
                             const map::Pose &>(&map::Map::CreateShot),
           py::arg("shot_id"), py::arg("shot_cam_id"),
           py::arg("pose") = map::Pose(),
           py::return_value_policy::reference_internal)
      .def("create_shot",
           py::overload_cast<const map::ShotId, const map::ShotCamera &,
                             const map::Pose &>(&map::Map::CreateShot),
           py::arg("shot_id"), py::arg("shot_cam"),
           py::arg("pose") = map::Pose(),
           py::return_value_policy::reference_internal)
      .def("create_shot",
           py::overload_cast<const map::ShotId, const std::string &,
                             const map::Pose &>(&map::Map::CreateShot),
           py::arg("shot_id"), py::arg("shot_cam"),
           py::arg("pose") = map::Pose(),
           py::return_value_policy::reference_internal)
      .def("update_shot_pose", &map::Map::UpdateShotPose)
      .def("remove_shot", &map::Map::RemoveShot)
      .def("get_shot", &map::Map::GetShot,
           py::return_value_policy::reference_internal)
      .def("add_observation",
           py::overload_cast<const map::ShotId, const map::LandmarkId,
                             const map::FeatureId>(&map::Map::AddObservation),
           py::arg("shot"), py::arg("landmark"), py::arg("feature_id"))
      .def("add_observation",
           py::overload_cast<map::Shot *const, map::Landmark *const,
                             const map::FeatureId>(&map::Map::AddObservation),
           py::arg("shot"), py::arg("landmark"), py::arg("feature_id"))
      .def("add_observation",
           py::overload_cast<map::Shot *const, map::Landmark *const,
                             const Observation &>(&map::Map::AddObservation),
           py::arg("shot"), py::arg("landmark"), py::arg("observation"))
      .def("remove_observation", &map::Map::RemoveObservation)
      .def("get_all_shots", &map::Map::GetAllShotPointers,
           py::return_value_policy::reference_internal)
      .def("get_all_cameras", &map::Map::GetAllCameraPointers,
           py::return_value_policy::reference_internal)
      .def("get_all_landmarks", &map::Map::GetAllLandmarkPointers,
           py::return_value_policy::reference_internal)
      .def("set_reference", &map::Map::SetTopoCentricConverter)
      .def("get_reference", &map::Map::GetTopoCentricConverter)
      .def("create_cam_model", &map::Map::CreateCameraModel,
           py::return_value_policy::reference_internal)
      .def("has_landmark", &map::Map::HasLandmark)
      .def("get_landmark", &map::Map::GetLandmark,
           py::return_value_policy::reference_internal)
      .def("clear_observations_and_landmarks",
           &map::Map::ClearObservationsAndLandmarks)
      .def("get_shot_camera", &map::Map::GetShotCamera,
           py::return_value_policy::reference_internal)
      .def("get_all_camera_models", &map::Map::GetAllCameraModels,
           py::return_value_policy::reference_internal)
      .def("get_camera_model", &map::Map::GetCameraModel,
           py::return_value_policy::reference_internal)

      ;

  py::class_<map::TopoCentricConverter>(m, "TopoCentriConverter")
      .def(py::init<>())
      .def(py::init<const double, const double, const double>())
      .def_readonly("lat", &map::TopoCentricConverter::lat_)
      .def_readonly("lon", &map::TopoCentricConverter::long_)
      .def_readonly("alt", &map::TopoCentricConverter::lat_);

  py::class_<map::Shot>(m, "Shot")
      .def(py::init<const map::ShotId, const map::ShotCamera &,
                    const map::Pose &>())
      .def_readonly("id", &map::Shot::id_)
      .def_readonly("unique_id", &map::Shot::unique_id_)
      .def_readonly("slam_data", &map::Shot::slam_data_,
                    py::return_value_policy::reference_internal)
      .def_readwrite("mesh", &map::Shot::mesh)
      .def("get_descriptor", &map::Shot::GetDescriptor,
           py::return_value_policy::reference_internal)
      .def("get_descriptors", &map::Shot::GetDescriptors,
           py::return_value_policy::reference_internal)
      .def("get_keypoint", &map::Shot::GetKeyPoint,
           py::return_value_policy::reference_internal)
      .def("get_keypoints", &map::Shot::GetKeyPoints,
           py::return_value_policy::reference_internal)
      .def("compute_num_valid_pts", &map::Shot::ComputeNumValidLandmarks)
      .def("get_valid_landmarks", &map::Shot::ComputeValidLandmarks,
           py::return_value_policy::reference_internal)
      .def("get_valid_landmarks_indices",
           &map::Shot::ComputeValidLandmarksIndices,
           py::return_value_policy::reference_internal)
      .def("get_valid_landmarks_and_indices",
           &map::Shot::ComputeValidLandmarksAndIndices,
           py::return_value_policy::reference_internal)
      .def("number_of_keypoints", &map::Shot::NumberOfKeyPoints)
      .def("init_and_take_datastructures",
           &map::Shot::InitAndTakeDatastructures)
      .def("init_keypts_and_descriptors", &map::Shot::InitKeyptsAndDescriptors)
      .def("undistort_keypts", &map::Shot::UndistortKeypts)
      .def("undistorted_keypts_to_bearings",
           &map::Shot::UndistortedKeyptsToBearings)
      .def("set_pose", &map::Shot::SetPose)
      .def("get_pose", &map::Shot::GetPose,
           py::return_value_policy::reference_internal)
      .def("compute_median_depth", &map::Shot::ComputeMedianDepthOfLandmarks)
      .def("scale_landmarks", &map::Shot::ScaleLandmarks)
      .def("scale_pose", &map::Shot::ScalePose)
      .def("remove_observation", &map::Shot::RemoveLandmarkObservation)
      .def("get_camera_to_world", &map::Shot::GetCamToWorld)
      .def("get_world_to_camera", &map::Shot::GetWorldToCam)
      // TODO: Move completely away from opencv
      .def("get_obs_by_idx", &map::Shot::GetKeyPointEigen)
      .def("get_camera_name", &map::Shot::GetCameraName)
      .def_readwrite("shot_measurement", &map::Shot::shot_measurements_)
      .def_readwrite("metadata", &map::Shot::shot_measurements_)
      .def_property("pose", &map::Shot::GetPose, &map::Shot::SetPose)
      .def_property_readonly("camera", &map::Shot::GetCameraModel,
                             py::return_value_policy::reference_internal)
      .def("create_observation", &map::Shot::CreateObservation)
      .def("get_landmark_observation", &map::Shot::GetLandmarkObservation,
           py::return_value_policy::reference_internal)
      .def("project", &map::Shot::Project);

  py::class_<map::SLAMShotData>(m, "SlamShotData")
      .def_readonly("undist_keypts", &map::SLAMShotData::undist_keypts_,
                    py::return_value_policy::reference_internal)
      .def("update_graph_node", &map::SLAMShotData::UpdateGraphNode);
  ;

  py::class_<map::SLAMLandmarkData>(m, "SlamLandmarkData")
      .def("get_observed_ratio", &map::SLAMLandmarkData::GetObservedRatio);

  py::class_<map::ShotMeasurements>(m, "ShotMeasurements")
      .def_readwrite("gps_dop", &map::ShotMeasurements::gps_dop_)
      .def_readwrite("gps_pos", &map::ShotMeasurements::gps_position_)
      .def_readwrite("gps_position", &map::ShotMeasurements::gps_position_)
      .def_readwrite("orientation", &map::ShotMeasurements::orientation_)
      .def_readwrite("capture_time", &map::ShotMeasurements::capture_time_)
      .def_readwrite("accelerometer", &map::ShotMeasurements::accelerometer)
      .def_readwrite("compass", &map::ShotMeasurements::compass)
      .def_readwrite("skey", &map::ShotMeasurements::skey);

  py::class_<map::ShotMesh>(m, "ShotMesh")
      .def_property("faces", &map::ShotMesh::GetFaces, &map::ShotMesh::SetFaces)
      .def_property("vertices", &map::ShotMesh::GetVertices,
                    &map::ShotMesh::SetVertices);

  py::class_<map::Landmark>(m, "Landmark")
      .def(py::init<const map::LandmarkId &, const Eigen::Vector3d &>())
      .def_readonly("id", &map::Landmark::id_)
      .def_readonly("unique_id", &map::Landmark::unique_id_)
      .def_readwrite("slam_data", &map::Landmark::slam_data_)
      .def("get_global_pos", &map::Landmark::GetGlobalPos)
      .def_property("coordinates", &map::Landmark::GetGlobalPos,
                    &map::Landmark::SetGlobalPos)
      .def("set_global_pos", &map::Landmark::SetGlobalPos)
      .def("is_observed_in_shot", &map::Landmark::IsObservedInShot)
      .def("add_observation", &map::Landmark::AddObservation)
      .def("remove_observation", &map::Landmark::RemoveObservation)
      .def("has_observations", &map::Landmark::HasObservations)
      .def("get_observations", &map::Landmark::GetObservations,
           py::return_value_policy::reference_internal)
      .def("number_of_observations", &map::Landmark::NumberOfObservations)
      .def("get_ref_shot", &map::Landmark::GetRefShot,
           py::return_value_policy::reference_internal)
      .def("set_ref_shot", &map::Landmark::SetRefShot)
      .def("get_obs_in_shot", &map::Landmark::GetObservationInShot)
      .def_property("reprojection_errors",
                    &map::Landmark::GetReprojectionErrors,
                    &map::Landmark::SetReprojectionErrors)
      .def("remove_reprojection_error", &map::Landmark::RemoveReprojectionError)
      .def_property("color", &map::Landmark::GetColor,
                    &map::Landmark::SetColor);

  py::class_<map::ShotCamera>(m, "ShotCamera")
      .def(py::init<map::Camera &, const map::CameraId, const std::string &>())
      .def_readonly("id", &map::ShotCamera::id_)
      .def_readonly("camera_name", &map::ShotCamera::camera_name_);

  py::class_<map::Camera, map::PyCamera>(m, "CameraModel")
      .def(py::init<const size_t, const size_t, const std::string &>(),
           py::arg("width"), py::arg("height"), py::arg("projection_type"))
      .def("pixel_bearing", &map::Camera::PixelBearing)
      .def_readonly("id", &map::Camera::id)
      .def_readonly("projection_type", &map::Camera::projectionType)
      .def_readonly("width", &map::Camera::width)
      .def_readonly("height", &map::Camera::height)
      .def("get_K", &map::Camera::GetK);

  py::class_<map::BrownPerspectiveCamera, map::Camera>(m,
                                                       "BrownPerspectiveCamera")
      .def(
          py::init<const size_t, const size_t, const std::string &, const float,
                   const float, const float, const float, const float,
                   const float, const float, const float, const float>(),
          py::arg("width"), py::arg("height"), py::arg("projection_type"),
          py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"),
          py::arg("k1"), py::arg("k2"), py::arg("p1"), py::arg("p2"),
          py::arg("k3"));

  py::class_<map::PerspectiveCamera, map::Camera>(m, "PerspectiveCamera")
      .def(py::init<const size_t, const size_t, const std::string &,
                    const float, const float, const float>(),
           py::arg("width"), py::arg("height"), py::arg("projection_type"),
           py::arg("focal"), py::arg("k1"), py::arg("k2"))
      .def_readwrite("focal", &map::PerspectiveCamera::focal)
      .def_readwrite("k1", &map::PerspectiveCamera::k1)
      .def_readwrite("k2", &map::PerspectiveCamera::k2);
}