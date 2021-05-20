#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <glog/logging.h>

#include <map/pose.h>
#include <map/defines.h>
#include <map/map.h>
#include <map/shot.h>
#include <map/landmark.h>
#include <map/camera.h>
#include <map/map_io.h>
#include <map/observation.h>
namespace py = pybind11;
PYBIND11_MODULE(pymap, m) {

  py::class_<map::Pose>(m, "Pose")
    .def(py::init())
    .def("get_cam_to_world", &map::Pose::CameraToWorld)
    .def("get_world_to_cam", &map::Pose::WorldToCamera)
    .def("set_from_cam_to_world", py::overload_cast<const Eigen::Matrix4d&>(&map::Pose::SetFromCameraToWorld))
    .def("set_from_cam_to_world", py::overload_cast<const Eigen::Matrix3d&, const Eigen::Vector3d&>(&map::Pose::SetFromCameraToWorld))
    .def("set_from_cam_to_world", py::overload_cast<const Eigen::Vector3d&, const Eigen::Vector3d&>(&map::Pose::SetFromCameraToWorld))
    .def("set_from_world_to_cam", py::overload_cast<const Eigen::Matrix4d&>(&map::Pose::SetFromWorldToCamera))
    .def("set_from_world_to_cam", py::overload_cast<const Eigen::Matrix3d&, const Eigen::Vector3d&>(&map::Pose::SetFromWorldToCamera))
    .def("set_from_world_to_cam", py::overload_cast<const Eigen::Vector3d&, const Eigen::Vector3d&>(&map::Pose::SetFromWorldToCamera))
    .def("get_origin", &map::Pose::GetOrigin)
    .def("get_R_cam_to_world", &map::Pose::RotationCameraToWorld)
    .def("get_R_world_to_cam", &map::Pose::RotationWorldToCamera)
    .def("get_R_cam_to_world_min", &map::Pose::RotationCameraToWorldMin)
    .def("get_R_world_to_cam_min", &map::Pose::RotationWorldToCameraMin)
    .def("get_t_cam_to_world", &map::Pose::TranslationCameraToWorld)
    .def("get_t_world_to_cam", &map::Pose::TranslationWorldToCamera)
  ;
  py::class_<map::MapIO>(m, "MapIO")
    .def("save_map", &map::MapIO::SaveMapToFile)
    .def("color_map", &map::MapIO::ColorMap)
  ;
  py::class_<map::Map>(m, "Map")
    .def(py::init())
    .def("number_of_shots", &map::Map::NumberOfShots, "Returns the number of shots")
    .def("number_of_landmarks", &map::Map::NumberOfLandmarks)
    .def("number_of_cameras", &map::Map::NumberOfCameras)
    .def("create_shot_camera", &map::Map::CreateShotCamera, 
      py::arg("cam_id"), 
      py::arg("camera"),
      py::arg("name") = "",
      py::return_value_policy::reference_internal)
    // .def("update_shot_camera", &map::Map::UpdateShotCamera)
    .def("remove_shot_camera", &map::Map::RemoveShotCamera)
    // Landmark
    .def("create_landmark", &map::Map::CreateLandmark,
        py::arg("lm_id"),
        py::arg("global_position"),
        py::arg("name") = "",
        py::return_value_policy::reference_internal)
    .def("update_landmark", &map::Map::UpdateLandmark)
    .def("remove_landmark", py::overload_cast<const map::Landmark* const>(&map::Map::RemoveLandmark))
    .def("remove_landmark", py::overload_cast<const map::LandmarkId>(&map::Map::RemoveLandmark))
    .def("next_unique_landmark_id", &map::Map::GetNextUniqueLandmarkId)
    // Shot
    .def("create_shot", 
      py::overload_cast<const map::ShotId, const map::CameraId,
                        const std::string&, const map::Pose&>(&map::Map::CreateShot),
      py::arg("shot_id"), 
      py::arg("shot_cam_id"),
      py::arg("name") = "",
      py::arg("pose") = map::Pose(), 
      py::return_value_policy::reference_internal)
    .def("create_shot", 
      py::overload_cast<const map::ShotId, const map::ShotCamera&,
                        const std::string&, const map::Pose&>(&map::Map::CreateShot),
      py::arg("shot_id"),
      py::arg("shot_cam"),
      py::arg("name") = "",
      py::arg("pose") = map::Pose(),
      py::return_value_policy::reference_internal)
    .def("update_shot_pose", &map::Map::UpdateShotPose)
    .def("remove_shot", &map::Map::RemoveShot)
    .def("next_unique_shot_id", &map::Map::GetNextUniqueShotId)
    .def("get_shot", &map::Map::GetShot, py::return_value_policy::reference_internal)

    .def("add_observation", &map::Map::AddObservation,
         py::arg("shot"), py::arg("landmark"), py::arg("feature_id"))
    .def("remove_observation", &map::Map::RemoveObservation)
    .def("get_all_shots", &map::Map::GetAllShotPointers, py::return_value_policy::reference_internal)
    .def("get_all_cameras", &map::Map::GetAllCameraPointers, py::return_value_policy::reference_internal)
    .def("get_all_landmarks", &map::Map::GetAllLandmarkPointers, py::return_value_policy::reference_internal)
  ;

  py::class_<map::Shot>(m, "Shot")
    .def(py::init<const map::ShotId, const map::ShotCamera&, 
                  const map::Pose&, const std::string&>())
    .def_readonly("id", &map::Shot::id_)
    .def_readonly("name", &map::Shot::name_)
    .def_readonly("slam_data", &map::Shot::slam_data_, py::return_value_policy::reference_internal)
    .def("get_descriptor", &map::Shot::GetDescriptor, py::return_value_policy::reference_internal)
    .def("get_descriptors", &map::Shot::GetDescriptors, py::return_value_policy::reference_internal)
    .def("get_keypoint", &map::Shot::GetKeyPoint, py::return_value_policy::reference_internal)
    .def("get_keypoints", &map::Shot::GetKeyPoints, py::return_value_policy::reference_internal)
    .def("compute_num_valid_pts", &map::Shot::ComputeNumValidLandmarks)
    .def("get_valid_landmarks", &map::Shot::ComputeValidLandmarks, py::return_value_policy::reference_internal)
    .def("get_valid_landmarks_indices", &map::Shot::ComputeValidLandmarksIndices, py::return_value_policy::reference_internal)
    .def("get_valid_landmarks_and_indices", &map::Shot::ComputeValidLandmarksAndIndices, py::return_value_policy::reference_internal)
    .def("number_of_keypoints", &map::Shot::NumberOfKeyPoints)
    .def("init_and_take_datastructures", &map::Shot::InitAndTakeDatastructures)
    .def("init_keypts_and_descriptors", &map::Shot::InitKeyptsAndDescriptors)
    .def("undistort_keypts", &map::Shot::UndistortKeypts)
    .def("undistorted_keypts_to_bearings", &map::Shot::UndistortedKeyptsToBearings)
    .def("set_pose", &map::Shot::SetPose)
    .def("get_pose", &map::Shot::GetPose, py::return_value_policy::reference_internal)
    .def("compute_median_depth", &map::Shot::ComputeMedianDepthOfLandmarks)
    .def("scale_landmarks", &map::Shot::ScaleLandmarks)
    .def("scale_pose", &map::Shot::ScalePose)
    .def("remove_observation", &map::Shot::RemoveLandmarkObservation)
    .def("get_camera_to_world", &map::Shot::GetCamToWorld)
    .def("get_world_to_camera", &map::Shot::GetWorldToCam)
    //TODO: Move completely away from opencv
    .def("get_obs_by_idx", &map::Shot::GetKeyPointEigen)
    // .def_readonly("descriptors", &map::Shot::descriptors_eig_)
  ;

  py::class_<map::SLAMShotData>(m, "SlamShotData")
    .def_readonly("undist_keypts", &map::SLAMShotData::undist_keypts_, py::return_value_policy::reference_internal)
    .def("update_graph_node", &map::SLAMShotData::UpdateGraphNode);
  ;

  py::class_<map::SLAMLandmarkData>(m , "SlamLandmarkData")
    .def("get_observed_ratio", &map::SLAMLandmarkData::GetObservedRatio)
    // .def_readonly("")
  ;

  py::class_<map::Landmark>(m, "Landmark")
    .def(py::init<const map::LandmarkId&, const Eigen::Vector3d&, const std::string&>())
    .def_readonly("id", &map::Landmark::id_)
    .def_readonly("name", &map::Landmark::name_)
    .def_readwrite("slam_data", &map::Landmark::slam_data_)
    .def("get_global_pos", &map::Landmark::GetGlobalPos)
    .def("set_global_pos", &map::Landmark::SetGlobalPos)
    .def("is_observed_in_shot", &map::Landmark::IsObservedInShot)
    .def("add_observation", &map::Landmark::AddObservation)
    .def("remove_observation", &map::Landmark::RemoveObservation)
    .def("has_observations", &map::Landmark::HasObservations)
    .def("get_observations", &map::Landmark::GetObservations, py::return_value_policy::reference_internal)
    .def("number_of_observations", &map::Landmark::NumberOfObservations)
    .def("get_ref_shot", &map::Landmark::GetRefShot, py::return_value_policy::reference_internal)
    .def("set_ref_shot", &map::Landmark::SetRefShot)
  ;

  py::class_<map::ShotCamera>(m, "ShotCamera")
    .def(py::init<const map::Camera&, const map::CameraId, const std::string&>())
    .def_readonly("id", &map::ShotCamera::id_)
    .def_readonly("camera_name", &map::ShotCamera::camera_name_)
  ;

  py::class_<map::Camera>(m, "Camera")
    .def(py::init<const size_t, const size_t, const std::string&>(),
         py::arg("width"), py::arg("height"), py::arg("projection_type"))
  ;

  py::class_<map::BrownPerspectiveCamera, map::Camera>(m, "BrownPerspectiveCamera")
    .def(py::init<const size_t, const size_t, const std::string&,
                  const float, const float, const float, const float,
                  const float, const float, const float, const float, const float>(),
                  py::arg("width"), py::arg("height"), py::arg("projection_type"),
                  py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"),
                  py::arg("k1"), py::arg("k2"), py::arg("p1"), py::arg("p2"), py::arg("k3"))
  ;

  py::class_<map::Observation>(m, "Observation")
    .def(py::init<double, double, double, int, int, int, int>())
    .def(py::init<double, double, double, int, int, int, int,
                  float, float, float, int>())
    .def_readwrite("point", &map::Observation::point)
    .def_readwrite("scale", &map::Observation::scale)
    .def_readwrite("id", &map::Observation::id)
    .def_readwrite("color", &map::Observation::color)
    .def_readwrite("angle", &map::Observation::angle)
    .def_readwrite("response", &map::Observation::response)
    .def_readwrite("size", &map::Observation::size)
    .def_readwrite("class_id", &map::Observation::class_id)
  ;
}
