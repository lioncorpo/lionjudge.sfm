from opensfm import pyslam
from opensfm import pymap
from opensfm import features
from opensfm import pybundle
from opensfm import reconstruction
from opensfm import types

from slam_mapper import SlamMapper

import slam_debug
import slam_utils
import logging
import numpy as np
logger = logging.getLogger(__name__)


class SlamTracker(object):
    def __init__(self, guided_matcher):
        print("init slam tracker")
        self.guided_matcher = guided_matcher
        self.scale_factors = None
        self.num_tracked_lms = 0

    def track(self, slam_mapper: SlamMapper, curr_shot: pymap.Shot, config, camera,
              data):
        """Tracks the current frame with respect to the reconstruction
        """

        """ last_shot, frame, camera, init_pose, config, data):
        Align the current frame to the already estimated landmarks
            (visible in the last frame)
            landmarks visible in last frame
        """

        # Try to match to last frame first
        init_pose = slam_mapper.last_shot.get_pose()
        print("init_pose: ", init_pose.get_cam_to_world())
        chrono = slam_debug.Chronometer()
        pose_tracking = self.track_motion(slam_mapper, curr_shot,
                                          camera, config, data)
        chrono.lap("track_motion")
        
        
        # Update local map!
        # local_keyframes = pyslam.SlamUtilities.update_local_keyframes(
            # curr_shot)
        # print("new lk: ", len(local_keyframes), " old_lk: ")
        # local_landmarks = pyslam.SlamUtilities.update_local_landmarks(
            # local_keyframes)
        # n_loc_kfs = pyslam.SlamUtilities.match_shot_to_local_lms(curr_shot, self.guided_matcher)
        # assert(len(local_keyframes) == n_loc_kfs)
        # print("len(local_keyframes) == n_loc_kfs",n_loc_kfs, len(local_keyframes))
        chrono.lap("update_local_landmarks")
        pose_tracking_2 = pymap.Pose()
        pose_tracking_2.set_from_world_to_cam(
            slam_utils.pose_to_mat(pose_tracking))
        curr_shot.set_pose(pose_tracking_2)
        # n_matches = self.guided_matcher.search_local_landmarks(
        #     local_landmarks, curr_shot)
        # TODO: REMOVE DEBUG VISUALIZATION
        slam_debug.check_shot_for_double_entries(curr_shot)
        n_valid_pts_bef = curr_shot.compute_num_valid_pts(1) # TODO: Remove debug stuff
        n_matches = pyslam.SlamUtilities.match_shot_to_local_lms(curr_shot, self.guided_matcher)
        n_valid_pts_aft = curr_shot.compute_num_valid_pts(1) # TODO: Remove debug stuff
        print("n_matches {} found in current frame. bef{} aft {}".format(n_matches, n_valid_pts_bef, n_valid_pts_aft))
        assert(n_matches + n_valid_pts_bef == n_valid_pts_aft)
        # TODO: REMOVE DEBUG VISUALIZATION
        slam_debug.check_shot_for_double_entries(curr_shot)  # TODO: Remove debug stuff
        # val_lms = curr_shot.get_valid_landmarks()
        # val_lms_idc = curr_shot.get_valid_landmarks_indices()
        # matches_last = []
        # for lm, new_id in zip(val_lms, val_lms_idc):
        #     obs = lm.get_observations()
        #     if slam_mapper.last_shot in obs:
        #         matches_last.append((obs[slam_mapper.last_shot], new_id))
        # TODO: REMOVE DEBUG STUFF
        # pts1 = pyslam.SlamUtilities.keypts_from_shot(slam_mapper.last_shot)
        # pts2 = pyslam.SlamUtilities.keypts_from_shot(curr_shot)
        # slam_debug.disable_debug = False
        #matches_last = np.asarray(matches_last)
        # slam_debug.visualize_matches_pts(pts1, pts2, matches_last, data.load_image(slam_mapper.last_shot.name), data.load_image(curr_shot.name),
        #                                  is_normalized=False, do_show=True)
        # TODO: REMOVE DEBUG STUFF
        chrono.lap("search_local_landmarks")
        # Now, local optimization
        lms = curr_shot.get_valid_landmarks()
        points2D = pyslam.SlamUtilities.get_valid_kpts_from_shot(curr_shot)
        valid_ids = curr_shot.get_valid_landmarks_indices()
        print("got: ", len(lms), " landmarks and ", len(points2D))
        chrono.start()
        points3D = np.zeros((len(lms), 3), dtype=np.float)
        for i, lm in enumerate(lms):
            points3D[i, :] = lm.get_global_pos()
        observations, _, _ = features.normalize_features(
            points2D, None, None, camera[1].width, camera[1].height)

        # TODO: Remove debug stuff
        # slam_debug.disable_debug = True
        # slam_debug.reproject_landmarks(points3D, observations,
        #                                slam_utils.pose_to_mat(pose_tracking),
        #                                data.load_image(curr_shot.name), camera[1],
        #                                title="bef tracking: "+curr_shot.name, obs_normalized=True, do_show=False)
        # slam_debug.avg_timings.addTimes(chrono.laps_dict)

        pose_init_sfm = slam_utils.mat_to_pose(curr_shot.get_pose().get_world_to_cam())
        pose, valid_pts = self.\
            bundle_tracking(points3D, observations, pose_init_sfm, camera, data.config, data)
        # print("pose: ", pose, " n_valid: ", len(valid_pts))
        # chrono.lap("track_local_map")
        # new_pose, n_valid = pyslam.SlamUtilities.bundle_tracking(curr_shot)
        # chrono.lap("new_bundle")
        # print(chrono.lap_times())
        # print("new_pose: ", new_pose, "valid: ", n_valid)
        # print("new_pose: ", new_pose.get_world_to_cam(), "vs", pose.get_Rt())
        # assert(np.allclose(new_pose.get_R_world_to_cam(), pose.get_rotation_matrix()))
        # exit(0)
        slam_debug.avg_timings.addTimes(chrono.laps_dict)

        slam_debug.reproject_landmarks(points3D, observations,
                                       slam_utils.pose_to_mat(pose), data.load_image(curr_shot.name), camera[1],
                                       title="aft tracking: "+curr_shot.name, obs_normalized=True, do_show=True)
        slam_debug.disable_debug = True
        slam_debug.avg_timings.addTimes(chrono.laps_dict)
        chrono.start()
        print("valid: ", curr_shot.compute_num_valid_pts(1))
        n_tracked = 0
        for idx, is_valid in enumerate(valid_pts):
            if not is_valid:
                curr_shot.remove_observation(valid_ids[idx])
            else:
                n_tracked += 1
        assert(curr_shot.compute_num_valid_pts(1) == np.sum(valid_pts)) # TODO: Remove debug stuff

        slam_debug.check_shot_for_double_entries(curr_shot) # TODO: Remove debug stuff
        self.num_tracked_lms = n_tracked
        chrono.lap("filter_outliers")
        return pose

    def track_motion(self, slam_mapper: SlamMapper, curr_shot: pymap.Shot,
                     camera, config, data):
        """Estimate 6 DOF world pose of frame
        Reproject the landmarks seen in the last frame
        to frame and estimate the relative 6 DOF motion between
        the two by minimizing the reprojection error.
        """
        print("track_motion: ", slam_mapper.last_shot.name, "<->",
              curr_shot.name)
        # TODO: Make an actual update on the closest frames in the map
        # For now, simply take the last 10 keyframes
        # return

        margin = 20
        init_shot = slam_mapper.pre_last
        last_shot = slam_mapper.last_shot
        # TODO: Use the velocity from the slam mapper!!
        # Be careful with velocity after init!
        # velocity = last_shot.world_pose.compose(
        #     init_shot.world_pose.inverse())
        # print("velocity: ", velocity.get_Rt())

        # WORLD POSE = T_CW
        # pose_init = velocity.compose(last_shot.world_pose)
        # print("pose_init: ", pose_init.get_Rt())
        # print("slam_mapper.velocity: ", slam_mapper.velocity,
            #   "last_shot",last_shot.get_pose().get_world_to_cam())
        T_init = slam_mapper.velocity.dot(last_shot.get_pose().get_world_to_cam())
        # print("T_init: ", T_init)
        # END velocity
        # TODO: REMOVE DEBUG VISUALIZATION

        kf = slam_mapper.keyframes[-1]
        lms = kf.get_valid_landmarks()
        points3D = np.zeros((len(lms), 3), dtype=np.float)
        for idx, lm in enumerate(lms):
            points3D[idx, :] = lm.get_global_pos()
        # slam_debug.disable_debug = False
        T_last = last_shot.get_pose().get_world_to_cam()
        slam_debug.reproject_landmarks(points3D, pyslam.SlamUtilities.keypts_from_shot(last_shot),
                                       T_last, data.load_image(last_shot.name), camera[1], title="init_last", obs_normalized=False, do_show=False)
        slam_debug.reproject_landmarks(points3D, pyslam.SlamUtilities.keypts_from_shot(curr_shot),
                                       T_init, data.load_image(curr_shot.name), camera[1], title="init", obs_normalized=False, do_show=False)
        slam_debug.disable_debug = True
        # TODO: REMOVE DEBUG VISUALIZATION


        pose_init = pymap.Pose()
        pose_init.set_from_world_to_cam(T_init)
        curr_shot.set_pose(pose_init)
        slam_debug.check_shot_for_double_entries(curr_shot)  # TODO: Remove debug stuff
        slam_debug.check_shot_for_double_entries(last_shot)  # TODO: Remove debug stuff
        n_matches = self.guided_matcher.assign_shot_landmarks_to_kpts_new(slam_mapper.last_shot, curr_shot, margin)
        print("found matches: ", n_matches)
        if n_matches < 10:  # not enough matches found, increase margin
            print("matches2: ", margin)
            exit()
            n_matches = self.guided_matcher.\
                assign_shot_landmarks_to_kpts(slam_mapper.last_shot, curr_shot, margin * 2)
            if n_matches < 10:
                logger.error("Tracking lost!!")
                exit()
        slam_debug.check_shot_for_double_entries(curr_shot) # TODO: Remove debug stuff

        lms = curr_shot.get_valid_landmarks()
        points2D = pyslam.SlamUtilities.get_valid_kpts_from_shot(curr_shot)
        valid_ids = curr_shot.get_valid_landmarks_indices()
        print("got: ", len(lms), " landmarks and ", len(points2D))

        # normalize
        points2D, _, _ = features.\
            normalize_features(points2D, None, None,
                               camera[1].width, camera[1].height)

        points3D = np.zeros((len(lms), 3), dtype=np.float)
        for i, lm in enumerate(lms):
            points3D[i, :] = lm.get_global_pos()
        pose_init_sfm = slam_utils.mat_to_pose(T_init)
        # Set up bundle adjustment problem
        pose, valid_pts = self.bundle_tracking(
            points3D, points2D, pose_init_sfm, camera, config, data)

        # TODO: REMOVE DEBUG VISUALIZATION
        kf = slam_mapper.keyframes[-1]
        lms = curr_shot.get_valid_landmarks()
        points2D = pyslam.SlamUtilities.get_valid_kpts_from_shot(curr_shot)
        points3D = np.zeros((len(lms), 3), dtype=np.float)
        for idx, lm in enumerate(lms):
            points3D[idx, :] = lm.get_global_pos()
        # slam_debug.disable_debug = False
        slam_debug.\
            reproject_landmarks(points3D, points2D,
                                slam_utils.pose_to_mat(pose),
                                data.load_image(curr_shot.name),
                                camera[1], title="reproj",
                                obs_normalized=False, do_show=True)
        slam_debug.disable_debug = True
        # TODO: REMOVE DEBUG VISUALIZATION

        # Remove outliers
        print("valid: ", curr_shot.compute_num_valid_pts(1))
        n_tracked = 0
        for idx, is_valid in enumerate(valid_pts):
            if not is_valid:
                curr_shot.remove_observation(valid_ids[idx])
            else:
                n_tracked += 1
        assert(curr_shot.compute_num_valid_pts(1) == np.sum(valid_pts)) # TODO: Remove debug stuff
        assert(curr_shot.compute_num_valid_pts(1) == n_tracked) # TODO: Remove debug stuff
        self.num_tracked_lms = n_tracked
        slam_debug.check_shot_for_double_entries(curr_shot) # TODO: Remove debug stuff
        if np.sum(valid_pts) < 10:
            logger.error("Tracking lost!!")
            # TODO: ROBUST MATCHING
            exit()
        slam_debug.visualize_tracked_lms(points2D[valid_pts, :], curr_shot, data)
        return pose

    def bundle_tracking(self, points3D, observations, init_pose, camera,
                        config, data):
        """Estimates the 6 DOF pose with respect to 3D points

        Reprojects 3D points to the image plane and minimizes the
        reprojection error to the corresponding observations to 
        find the relative motion.

        Args:
            points3D: 3D points to reproject
            observations: their 2D correspondences
            init_pose: initial pose depending on the coord. system of points3D
            camera: intrinsic camera parameters
            config, data

        Returns:
            pose: The estimated (relative) 6 DOF pose
        """
        if len(points3D) != len(observations):
            print("len(points3D) != len(observations): ",
                  len(points3D), len(observations))
            return None
        # reproject_landmarks(points3D, observations, init_pose, camera, data)
        # match "last frame" to "current frame"
        # last frame could be reference frame
        # somehow match world points/landmarks seen in last frame
        # to feature matches
        fix_cameras = True
        chrono = slam_debug.Chronometer()

        ba = pybundle.BundleAdjuster()
        # for camera in reconstruction.cameras.values():
        reconstruction.\
            _add_camera_to_bundle(ba, camera[1], camera[1], fix_cameras)

        # constant motion velocity -> just say id
        shot_id = str(0)
        camera_id = str(camera[0])
        camera_const = False
        ba.add_shot(shot_id, str(camera_id), init_pose.rotation,
                    init_pose.translation, camera_const)
        points_3D_constant = True
        # Add points in world coordinates
        for (pt_id, pt_coord) in enumerate(points3D):
            ba.add_point(str(pt_id), pt_coord, points_3D_constant)
            ft = observations[pt_id, :]
            ba.add_point_projection_observation(shot_id, str(pt_id),
                                                ft[0], ft[1], ft[2])
        # Assume observations N x 3 (x,y,s)
        ba.add_absolute_up_vector(shot_id, [0, 0, -1], 1e-3)
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
        ba.set_max_num_iterations(50)
        ba.set_linear_solver_type("SPARSE_SCHUR")
        chrono.lap('setup')
        ba.run()
        chrono.lap('run_track')
        print("Tracking report: ", ba.full_report())
        s = ba.get_shot(shot_id)
        pose = types.Pose()
        pose.rotation = [s.r[0], s.r[1], s.r[2]]
        pose.translation = [s.t[0], s.t[1], s.t[2]]
        valid_pts = self.discard_outliers(ba, len(points3D), pose, camera[1])
        chrono.lap('discard_outliers')
        print(chrono.lap_times())
        # print("valid_pts!: ", valid_pts)
        return pose, valid_pts

    def discard_outliers(self, ba, n_pts, pose, camera):
        """Remove landmarks with large reprojection error
        or if reprojections are out of bounds
        """
        pts_outside = 0
        pts_inside = 0
        pts_outside_new = 0
        th = 0.006
        valid_pts = np.zeros(n_pts, dtype=bool)
        w, h = camera.width, camera.height
        for pt_id in range(0, n_pts):
            p = ba.get_point(str(pt_id))
            error = p.reprojection_errors['0']
            # Discard if reprojection error too large
            if np.linalg.norm(error) > th:
                pts_outside += 1
            else:
                # check if OOB
                camera_point = pose.transform([p.p[0], p.p[1], p.p[2]])
                if camera_point[2] <= 0.0:
                    pts_outside += 1
                    pts_outside_new += 1
                    continue
                point2D = camera.project(camera_point)
                if slam_utils.in_image(point2D, w, h):
                    pts_inside += 1
                    valid_pts[pt_id] = True
                else:
                    pts_outside += 1
                    pts_outside_new += 1

        print("pts inside {} and outside {}/ {}".
              format(pts_inside, pts_outside, pts_outside_new))
        return valid_pts
