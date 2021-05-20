import datetime
import logging
import numpy as np
import cv2
from opensfm import types
from opensfm import csfm
from opensfm.reconstruction import Chronometer
from opensfm import reconstruction
from opensfm import feature_loader
from opensfm import features

# from slam_matcher import SlamMatcher
import slam_matcher
from slam_mapper import SlamMapper
from slam_types import Frame
import slam_utils
import slam_debug
from itertools import compress
logger = logging.getLogger(__name__)

import cslam


class SlamTracker(object):

    # def __init__(self, data, config, guided_matcher):
    def __init__(self, guided_matcher):
        print("init slam tracker")
        self.guided_matcher = guided_matcher
        self.scale_factors = None
        self.num_tracked_lms = 0

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
        chrono = Chronometer()
        ba = csfm.BundleAdjuster()
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
        # print("Tracking report: ", ba.full_report())
        s = ba.get_shot(shot_id)
        pose = types.Pose()
        pose.rotation = [s.r[0], s.r[1], s.r[2]]
        pose.translation = [s.t[0], s.t[1], s.t[2]]
        valid_pts = self.discard_outliers(ba, len(points3D), pose, camera[1])
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

    def track_motion(self, slam_mapper: SlamMapper, frame: Frame,
                     camera, config, data):
        """Estimate 6 DOF world pose of frame
        Reproject the landmarks seen in the last frame
        to frame and estimate the relative 6 DOF motion between
        the two by minimizing the reprojection error.
        """
        print("track_motion: ", slam_mapper.last_frame.im_name, "<->",
              frame.im_name)
        # TODO: Make an actual update on the closest frames in the map
        # For now, simply take the last 10 keyframes
        # return

        margin = 20
        init_frame = slam_mapper.pre_last
        last_frame = slam_mapper.last_frame
        velocity = last_frame.world_pose.compose(
            init_frame.world_pose.inverse())
        print("velocity: ", velocity.get_Rt())

        # WORLD POSE = T_CW
        pose_init = velocity.compose(last_frame.world_pose)
        print("pose_init: ", pose_init.get_Rt())
        # test something
        kf = slam_mapper.c_keyframes[-1]
        lms = kf.get_valid_lms()
        frame.cframe.getKptsPy()
        points3D = np.zeros((len(lms), 3), dtype=np.float)
        for idx, lm in enumerate(lms):
            points3D[idx, :] = lm.get_pos_in_world()
        slam_debug.disable_debug = True
        slam_debug.reproject_landmarks(points3D, None,
                                       last_frame.world_pose, data.load_image(last_frame.cframe.im_name), camera[1], title="lf world", obs_normalized=False, do_show=False)
        # slam_debug.reproject_landmarks(points3D, None,
        slam_debug.reproject_landmarks(points3D, frame.cframe.getKptsPy(),
                                       pose_init, data.load_image(frame.cframe.im_name), camera[1], title="init", obs_normalized=False, do_show=True)
        slam_debug.disable_debug = True
        # End test something

        frame.cframe.set_Tcw(
            np.vstack((pose_init.get_Rt(), np.array([0, 0, 0, 1]))))
        print("frame.cframe: ", frame.cframe.get_Tcw(),
              " inv: ", frame.cframe.get_Twc())
        matches = self.guided_matcher.\
            match_current_and_last_frame(
                frame.cframe, slam_mapper.last_frame.cframe, margin)
        # im1, im2 = data.load_image(last_frame.im_name), data.load_image(frame.im_name)
        n_matches = len(matches)
        print("matches: ", n_matches)
        if n_matches < 10:  # not enough matches found, increase margin
            print("matches2: ", margin)
            matches = self.guided_matcher.\
                match_current_and_last_frame(
                    frame.cframe, slam_mapper.last_frame.cframe, margin * 2)
            if n_matches < 10:
                logger.error("Tracking lost!!")
                exit()

        lms = frame.cframe.get_valid_lms()
        points2D = frame.cframe.get_valid_keypts()
        valid_ids = frame.cframe.get_valid_idx()
        print("got: ", len(lms), " landmarks and ", len(points2D))

        # normalize
        points2D, _, _ = features.\
            normalize_features(points2D, None, None,
                               camera[1].width, camera[1].height)

        points3D = np.zeros((len(lms), 3), dtype=np.float)
        for i, lm in enumerate(lms):
            points3D[i, :] = lm.get_pos_in_world()

        # Set up bundle adjustment problem
        pose, valid_pts = self.bundle_tracking(
            points3D, points2D, pose_init, camera, config, data)

        # # test something
        # kf = slam_mapper.c_keyframes[-1]
        # lms = kf.get_valid_lms()
        # # points2D = kf.get_valid_keypts()
        # frame.cframe.getKptsPy()
        # points3D = np.zeros((len(lms), 3), dtype=np.float)
        # for idx, lm in enumerate(lms):
        #     points3D[idx, :] = lm.get_pos_in_world()
        # slam_debug.disable_debug = True
        # slam_debug.reproject_landmarks(points3D, frame.cframe.getKptsPy(),
        #     pose, data.load_image(frame.cframe.im_name), camera[1], title="repro",obs_normalized=False, do_show=True)
        # slam_debug.disable_debug = True
        # # End test something
        valid_idx_dbg = frame.cframe.get_valid_idx()
        frame.cframe.set_outlier(np.array(valid_ids)[np.invert(valid_pts)])
        # discard outlier matches
        num_valid_matches = frame.cframe.discard_outliers()
        valid_idx_dbg_2 = frame.cframe.get_valid_idx()
        print("valid bef: ", len(valid_idx_dbg),
              " and after: ", len(valid_idx_dbg_2))
        print("Tracked ", frame.cframe.im_name, " with pose ", pose.get_Rt())
        if num_valid_matches < 10:
            logger.error("Tracking lost!!")
            # Make previous frame to keyframe!
            self.robust_matching(slam_mapper)
            exit()
        slam_debug.visualize_tracked_lms(points2D[valid_pts, :], frame, data)
        return pose

    def robust_matching(self, slam_mapper):  # , curr_frame, ref_frame):
        """Tracks the current frame with respect to ref_frame (either last frame or last kf)
        by exhaustive matching!
        Search for matches, estimate new pose and add this frame as KF
        """
        # First match to
        # matches = self.guided_matcher.match_frame_to_frame_exhaustive(old)
        # Match to last KF
        # match_keyframe_to_frame_exhaustive
        pass

    def track(self, slam_mapper: SlamMapper, frame: Frame, config, camera,
              data):
        """Tracks the current frame with respect to the reconstruction
        """

        """ last_frame, frame, camera, init_pose, config, data):
        Align the current frame to the already estimated landmarks
            (visible in the last frame)
            landmarks visible in last frame
        """

        # Try to match to last frame first
        init_pose = slam_mapper.last_frame.world_pose
        chrono = reconstruction.Chronometer()
        pose_tracking = self.track_motion(slam_mapper, frame,
                                          camera, config, data)
        chrono.lap("track_motion")
        # Update local map!
        local_keyframes = cslam.SlamUtilities.update_local_keyframes(
            frame.cframe)
        print("new lk: ", len(local_keyframes), " old_lk: ",
              len(slam_mapper.c_keyframes[-10:]))
        # local_landmarks = cslam.SlamUtilities.update_local_landmarks(slam_mapper.c_keyframes[-10:], frame.frame_id)
        local_landmarks = cslam.SlamUtilities.update_local_landmarks(
            local_keyframes, frame.frame_id)
        # print("new lms: ", len(local_lms), " old lms: ", len(local_landmarks))
        chrono.lap("update_local_landmarks")
        frame.cframe.set_Tcw(np.vstack((pose_tracking.get_Rt(), [0, 0, 0, 1])))
        n_matches = self.guided_matcher.search_local_landmarks(
            local_landmarks, frame.cframe)
        chrono.lap("search_local_landmarks")
        # Now, local optimization
        print("n_matches {} found in current frame.".format(n_matches))
        idx = 0
        valid_lms = frame.cframe.get_valid_lms()
        valid_kps = frame.cframe.get_valid_keypts()
        valid_ids = frame.cframe.get_valid_idx()
        points3D = np.zeros((len(valid_lms), 3))
        print("n_matches: ", n_matches, " len: ", len(valid_lms))
        for idx, lm in enumerate(valid_lms):
            points3D[idx, :] = lm.get_pos_in_world()
        observations, _, _ = features.normalize_features(
            valid_kps, None, None, camera[1].width, camera[1].height)

        # TODO: Remove debug stuff
        # slam_debug.disable_debug = False
        # slam_debug.reproject_landmarks(points3D, observations, pose_tracking,
        #                                frame.image, camera[1],
        #                                title="bef tracking: "+frame.im_name, obs_normalized=True, do_show=False)
        slam_debug.avg_timings.addTimes(chrono.laps_dict)
        chrono.start()
        pose, valid_pts = self.\
            bundle_tracking(points3D, observations, slam_utils.mat_to_pose(
                frame.cframe.get_Tcw()), camera, data.config, data)
        chrono.lap("track_local_map")
        slam_debug.avg_timings.addTimes(chrono.laps_dict)

        # print("pose after! ", pose.rotation, pose.translation)
        # print("valid_pts: ", len(valid_pts), " vs ", len(observations))
        # slam_debug.reproject_landmarks(points3D, observations,
        #                                pose, frame.image, camera[1],
        #                                title="aft tracking: "+frame.im_name, obs_normalized=True, do_show=True)
        # slam_debug.disable_debug = True

        # for

        slam_debug.avg_timings.addTimes(chrono.laps_dict)
        chrono.start()
        self.num_tracked_lms = np.sum(valid_pts)
        frame.cframe.set_outlier(np.array(valid_ids)[np.invert(valid_pts)])
        n_tracked = frame.cframe.clean_and_tick_landmarks()
        chrono.lap("filter_outliers")
        slam_debug.avg_timings.addTimes(chrono.laps_dict)
        print("n tracked: ", n_tracked)
        return pose
