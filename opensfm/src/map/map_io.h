#pragma once
#include <third_party/json/json.hpp>
using json = nlohmann::json;

namespace map
{
class Map;
class Shot;
class ShotCamera;
class Landmark;

class MapIO
{
public:
  static void SaveMapToFile(const Map& rec_map, const std::string& path);
  static void ColorMap(Map& rec_map);
  static json MapToJson(const Map& rec_map);
  static json ShotToJson(const Shot& shot);
  static json LandmarkToJson(const Landmark& landmark);
  static json CameraToJson(const ShotCamera& camera);

};
}