#include <map/map_io.h>
#include <map/shot.h>
#include <map/map.h>
#include <map/landmark.h>
#include <map/camera.h>
#include <fstream>
#include <iomanip>

using json = nlohmann::json;

namespace map
{
void MapIO::SaveMapToFile(const Map& rec_map, const std::string& path)
{
  std::ofstream out(path);
  out << std::setw(4) << MapToJson(rec_map) << std::endl;
}

json MapIO::MapToJson(const Map& rec_map)
{
  json map_json;
  // Add cameras
  for (const auto& id_and_cam : rec_map.GetAllCameras())
  {
    map_json["cameras"][id_and_cam.second->camera_name_] = CameraToJson(*id_and_cam.second);
  }
  // Add shots
  for (const auto& id_and_shot : rec_map.GetAllShots())
  {
    map_json["shots"][id_and_shot.second->id_] = ShotToJson(*id_and_shot.second);
  }

  // Add landmarks
  for (const auto& id_and_lm : rec_map.GetAllLandmarks())
  {
    map_json["points"][id_and_lm.first] = LandmarkToJson(*id_and_lm.second);
  }

  return map_json;
}

json MapIO::ShotToJson(const Shot &shot)
{
  const auto& pose = shot.GetPose();
  const Eigen::Vector3d r_cw = pose.RotationWorldToCameraMin();
  const Eigen::Vector3d t_cw = pose.TranslationWorldToCamera();
  // TODO: Write the metadata like in io.py
  // orientation, capture_time, gps_dop...
  return {
          {"rotation", {r_cw[0], r_cw[1], r_cw[2]} },
          {"translation", {t_cw[0], t_cw[1], t_cw[2]}},
          {"camera",shot.shot_camera_.camera_name_}
        };
}

json MapIO::LandmarkToJson(const Landmark &landmark)
{
  const Eigen::Vector3i& color = landmark.GetColor();
  const Eigen::Vector3d& pos = landmark.GetGlobalPos();
  return {
          {"color", {color[0], color[1], color[2]} },
          {"coordinates", {pos[0], pos[1], pos[2]} },
         };
}

json MapIO::CameraToJson(const ShotCamera& camera)
{
  json cam_json;
  const auto& cam_model = camera.camera_model_;
  // not specific to any camera
  cam_json["projection_type"] = cam_model.projectionType;
  cam_json["width"] = cam_model.width;
  cam_json["height"] = cam_model.height;

  if (cam_model.projectionType.compare("perspective") == 0)
  {
    const PerspectiveCamera* const cam = dynamic_cast<const PerspectiveCamera*>(&cam_model);

    cam_json["focal"] = cam->focal;
    cam_json["k1"] = cam->k1;
    cam_json["k2"] = cam->k2;
    return cam_json;
  }
  else if (cam_model.projectionType.compare("brown") == 0)
  {
    const BrownPerspectiveCamera* const cam = dynamic_cast<const BrownPerspectiveCamera*>(&cam_model);
    cam_json["focal_x"] = cam->fx;
    cam_json["focal_y"] = cam->fy;
    cam_json["c_x"] = cam->cx;
    cam_json["c_y"] = cam->cy;
    cam_json["k1"] = cam->k1;
    cam_json["k2"] = cam->k2;
    cam_json["p1"] = cam->p1;
    cam_json["p2"] = cam->p2;
    cam_json["k3"] = cam->k3;
    return cam_json;
  }
  else if (cam_model.projectionType.compare("fisheye") == 0)
  {
    return cam_json;
  }
  else if (cam_model.projectionType.compare("dual") == 0)
  {
    return cam_json;
  }
  else if (cam_model.projectionType.compare("equirectangular") == 0 || cam_model.projectionType.compare("spherical") == 0)
  {
    return cam_json;
  }
  return cam_json;
}

void 
MapIO::ColorMap(Map& rec_map)
{
  //color all the landmarks

  for (auto& lm_id : rec_map.GetAllLandmarks())
  {
    auto& lm = lm_id.second;
    //get the first observation
    const auto& shot_obs = lm->GetObservations();
    const auto& first_obs_pair = shot_obs.cbegin();
    auto* first_shot = (*first_obs_pair).first;
    auto feat_id = (*first_obs_pair).second;
    const auto& first_obs = first_shot->GetObservation(shot_obs.at(first_shot));
    lm->SetColor(first_obs.color);
  }
}
} // namespace map