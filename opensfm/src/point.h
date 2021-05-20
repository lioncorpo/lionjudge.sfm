#pragma once

#include <Eigen/Eigen>
#include <opencv2/features2d/features2d.hpp>

#include <unordered_map>
#include <memory>

using ShotId = size_t;
using PointId = size_t;
using FeatureId = size_t;
using CameraId = size_t;

struct KeyCompare
{
    template<typename T>
    bool operator()(T* lhs, T* rhs) const { return lhs->kf_id_ < rhs->kf_id_; }
    bool operator()(T const* lhs, T const * rhs) const { return lhs->kf_id_ < rhs->kf_id_; }
};
struct SLAMPointData{
};
class Point {
 public:
  Point(const PointId point_id, const Eigen::Vector3d& global_pos, const std::string& name = "");

  Eigen::Vector3d GetGlobalPos() const { return global_pos_; }
  void SetGlobalPos(const Eigen::Vector3d& global_pos) const { global_pos_ = global_pos; }

  bool IsObservedInShot(Shot* shot) const;
  void AddObservation(Shot* shot, const FeatureId feat_id);
  void RemoveObservation(Shot* shot);
  void HasObservations() const

public:
  //We could set the const values to public, to avoid writing a getter.
  const std::string point_name_;
  const PointId id_;
private:
  Eigen::Vector3d global_pos_; // point in global
  std::map<Shot *, FeatureId, KeyCompare<Shot>> observations_;
  SLAMPointData slam_data_;
};

struct SLAMShotData{
};

class Shot {
 public:
  Shot(const ShotId shot_id, const camera* camera, const Pose& pose, const std::string& name = "");
  const cv::Mat& GetDescriptor(const FeatureId id) const { return descriptors_.row(id); }
  const cv::KeyPoint& GetKeyPoint(const FeatureId id) const { return keypoints_.at(id); }
  //No reason to set individual keypoints or descriptors

  //read-only access
  const std::vector<cv::KeyPoint>& GetKeyPoints() const { return keypoints_; }
  const cv::Mat& GetDescriptors() const { return descriptors_; }
  
  size_t ComputeNumValidPoints() const;
  
  const std::vector<Point*>& GetPoints() const { return points_; }
  std::vector<Point*>& GetPoints() { return points_; }
  void RemovePointObservation(const FeatureId id);
  void AddPointObservation(const Point* point, const FeatureId feat_id);
  void SetPose(const Pose& pose);
  SLAMShotData slam_data_;


 private:
  //We could set the const values to public, to avoid writing a getter.
  const std::string image_name_;
  const ShotId id_;

  std::vector<Point*> points_;
  std::vector<cv::KeyPoint> keypoints_;
  cv::Mat descriptors_;
  std::map<FeatureId, PointId> observations_;

  const ShotCamera *camera_;
  Pose pose_;

  ShotMeasurements shot_measurements_;

};


struct ShotMeasurements{
  Eigen::Vector3d gps_;
  double timestamp_;
};

class ShotCamera {
  Camera camera_;
  const int id_;
  const std::string camera_name_;
};

class Pose {
public:
  //4x4 Transformation
  Eigen::Matrix4d WorldToCamera() const { return worldToCam; }
  Eigen::Matrix4d CameraToWorld() const { return camToWorld; }

  // 3x3 Rotation
  Eigen::Matrix3d RotationWorldToCamera() const { return worldToCam.block<3,3>(0,0); }
  Eigen::Matrix3d RotationCameraToWorld() const { return camToWorld.block<3,3>(0,0); }
  
  // 3x1 Translation
  Eigen::Vector3d TranslationWorldToCamera() const { return worldToCam.block<3,1>(0,3); }
  Eigen::Vector3d TranslationCameraToWorld() const { return camToWorld.block<3,1>(0,3); };
  Eigen::Vector3d GetOrigin() const { return TranslationCameraToWorld(); }

  void SetPose(const Pose& pose);
private:
  // Eigen::Vector3d translation_;
  // Eigen::Vector3d rotation_;
  Eigen::Matrix4d worldToCam_;
  Eigen::Matrix4d camToWorld_;
  //Maybe use Sophus to store the minimum representation
};

class ReconstructionManager {
public:

  // Should belong to the manager
  ShotId GetShotIdFromName(const std::string& name) const { return shot_names_[name]; }
  PointId GetPointIdFromName(const std::string& name) const { return point_names_[name]; };

  ShotCamera* CreateCamera(const CameraId cam_id, const Camera& camera);
  void UpdateCamera(const CameraId cam_id, const Camera& camera);

  Shot* CreateShot(const ShotId shot_id, const CameraId camera_id, const Pose& pose, const std::string& name = "");
  void UpdateShotPose(const ShotId shot_id, const Pose& pose);

  Point* CreatePoint(const PointId point_id, const Eigen::Vector3d& global_pos, const std::string& name = "");
  void UpdatePoint(const PointId point_id, const Eigen::Vector3d& global_pos);
  bool AddObservation(const Shot* shot, const Point* point, const FeatureId feat_id);
  bool RemoveObservation(const Shot* shot, const Point* point, const FeatureId feat_id);

  std::map<Point*, FeatureId> GetObservationsOfShot(const Shot* shot);
  std::map<Shot*, FeatureId> GetObservationsOfPoint(const Point* point);  

  const std::unordered_map<ShotId, std::unique_ptr<Shot>>& GetAllShots() const { return shots_; }
  const std::unordered_map<int, std::unique_ptr<ShotCamera>>& GetAllCameras() const { return cameras_; };
  const std::unordered_map<PointId, std::unique_ptr<Point>>& GetAllPoints() const { return points_; };
private:
  //why no ponters?
  std::unordered_map<CameraId, std::unique_ptr<ShotCamera> > cameras_;
  std::unordered_map<ShotId, std::unique_ptr<Shot> > shots_;
  std::unordered_map<PointId, std::unique_ptr<Point> > points_;

  std::unordered_map<std::string, ShotId> shot_names_;
  std::unordered_map< std::string, PointId> point_names;
  // Alternatively, store pointer
  // std::unordered_map<std::string, Shot*> shot_names_;
  // std::unordered_map< std::string, Point*> point_names_;
};


using TrackId = int;

class TracksManager {
  public:
  std::vector<ShotId> GetShotIds();
  std::vector<TrackId> GetTrackIds();

  cv::KeyPoint GetObservation(const ShotId &shot, const TrackId &point);

  // Not sure if we use that
  std::map<PointId, cv::KeyPoint> GetObservationsOfShot(const ShotId& shot);

  // For point triangulation
  std::map<ShotId, cv::KeyPoint> GetObservationsOfPoint(const PointId& point);

  // For shot resection
  std::map<PointId, cv::KeyPoint> GetObservationsOfPointsAtShot(const std::vector<PointId>& points, const ShotId& shot);

  // For pair bootstrapping
  using ShotIdPair = std::pair<ShotId, ShotId>;
  using KeyPointPair = std::pair<cv::KeyPoint, cv::KeyPoint>;
  std::map<ShotIdPair, KeyPointPair>
  GetAllCommonObservations(const ShotId& shot1, const ShotId& shot2);
};


void do_bundle_in_python(metadata, manager) {

    ba = pybundle.BundleAdjuster()

    for camera in manager.GetAllCameras():
        camera_prior = camera_priors[camera.id]
        _add_camera_to_bundle(ba, camera, camera_prior, fix_cameras)

    for shot in manager.GetAllShots():
        // r = shot.pose.rotation ???
        // t = shot.pose.translation ???
        ba.add_shot(shot.id, shot.camera.id, r, t, False)

    for point in manager.GetAllPoints():
        ba.add_point(point.id, point.coordinates, False)

    for shot in manager.GetAllShots():
      for obs in shot.GetObservations():
        point = obs.feature.keypoint
        scale = obs.feature.scale
        ba.add_point_projection_observation(
            shot.id, obs.point->id, point[0], point[1], scale)

    ///// ??? WHERE DO WE PUT THE METADATA ???
    //// ??? Remove it from Shot in let it be passed
    //// a as map like camera_priors ???
    if config['bundle_use_gps']:
        for shot in manager.GetAllShots():
          shot_metadata = metadata[shot.id]
          g = shot_metadata.gps_position /// ??? DIRECT ACCESS ? Or a Get ?
          ba.add_position_prior(shot.id, g[0], g[1], g[2],
                                shot_metadata.gps_dop)

    if config['bundle_use_gcp'] and gcp:
        _add_gcp_to_bundle(ba, gcp, reconstruction.shots)

    align_method = config['align_method']
    if align_method == 'auto':
        align_method = align.detect_alignment_constraints(config, reconstruction, gcp)
    if align_method == 'orientation_prior':
        if config['align_orientation_prior'] == 'vertical':
            for shot_id in reconstruction.shots:
                ba.add_absolute_up_vector(shot_id, [0, 0, -1], 1e-3)
        if config['align_orientation_prior'] == 'horizontal':
            for shot_id in reconstruction.shots:
                ba.add_absolute_up_vector(shot_id, [0, -1, 0], 1e-3)

    ba.set_point_projection_loss_function(config['loss_function'],
                                          config['loss_function_threshold'])
    ba.set_internal_parameters_prior_sd(
        config['exif_focal_sd'],
        config['principal_point_sd'],
        config['radial_distorsion_k1_sd'],
        config['radial_distorsion_k2_sd'],
        config['radial_distorsion_p1_sd'],
        config['radial_distorsion_p2_sd'],
        config['radial_distorsion_k3_sd'])
    ba.set_num_threads(config['processes'])
    ba.set_max_num_iterations(config['bundle_max_iterations'])
    ba.set_linear_solver_type("SPARSE_SCHUR")

    chrono.lap('setup')
    ba.run()
    chrono.lap('run')

    for camera in manager.GetAllCameras():
      c = ba.get_perspective_camera(camera.id)
      camera_geometry = PerspectiveCamera(c.focal, c.k1, c.k2);
      manager.UpdateCamera(camera.id, camera_geometry);

    for shot in manager.GetAllShots():
        s = ba.get_shot(shot.id)
        manager.UpdateShotPose(camera.id,
                                [s.t[0], s.t[1], s.t[2]],
                                [s.r[0], s.r[1], s.r[2]]);

    for point in manager.GetAllPoints():
        p = ba.get_point(point.id)
        ///// ???? WHAT ABOUT REPROJECTION ERRORS ??? ////
        manager.UpdatePoint(camera.id, [p.p[0], p.p[1], p.p[2]], p.reprojection_errors)

    chrono.lap('teardown')

    logger.debug(ba.brief_report())
    report = {
        'wall_times': dict(chrono.lap_times()),
        'brief_report': ba.brief_report(),
    }
    return report
