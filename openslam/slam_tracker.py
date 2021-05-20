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


class SlamTracker(object):

    def __init__(self,  data, config):
        # self.slam_matcher = SlamMatcher(config)
        print("init slam tracker")

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
        reconstruction._add_camera_to_bundle(ba, camera[1], fix_cameras)
        
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
                     init_pose, camera, config, data):
        """Estimate 6 DOF world pose of frame 
        
        Reproject the landmarks seen in the last frame
        to frame and estimate the relative 6 DOF motion between
        the two by minimizing the reprojection error.
        """
        print("track_motion: ", slam_mapper.last_frame.im_name, "<->",
              frame.im_name, len(slam_mapper.last_frame.landmarks_),
              len(frame.landmarks_))
        
        chrono = Chronometer()
        margin = 10
        matches = slam_matcher.\
            match_frame_to_landmarks(frame.descriptors,
                                     slam_mapper.last_frame.landmarks_,
                                     margin, data, slam_mapper.graph)
        chrono.lap('matching')
        matches_test = slam_matcher.match_desc_desc(frame.descriptors, 
                                                    slam_mapper.curr_kf.descriptors, data)
        print("matches_test: ", matches_test)
        # if len(matches) < 30:
        #     print("Not enough matches!", len(matches))
        #     return None
        matches = np.asarray(matches)
        print("track_motion matches: ", matches.shape)
        landmarks = [slam_mapper.last_frame.landmarks_[m1] for m1 in matches[:, 1]]
        print("n_matches: ", len(matches))
        points3D = np.zeros((len(landmarks), 3))
        points = slam_mapper.reconstruction.points
        for idx, lm_id in enumerate(landmarks):
            points3D[idx, :] = points[lm_id].coordinates
            frame.landmarks_.append(lm_id)
        print("len(last_frame.landmarks_): ",
              len(slam_mapper.last_frame.landmarks_),
              "len(landmarks): ", len(frame.landmarks_),
              "len(points3D): ", len(points3D))
        points2D, _, _ = frame.load_points_desc_colors()
        print("points2D.shape: ", points2D.shape)
        points2D = points2D[matches[:, 0], :]
        chrono.lap("dummy")
        # Set up bundle adjustment problem
        pose, valid_pts = self.bundle_tracking(points3D, points2D, init_pose,
                                               camera, config, data)
        chrono.lap("tracking")
        print("tracker f2: ", frame.im_name, len(frame.landmarks_))
        frame.landmarks_ = list(compress(frame.landmarks_, valid_pts))
        print("tracker f3: ", frame.im_name, len(frame.landmarks_))
        print("Tracking times: ", chrono.lap_times())
        slam_debug.\
            visualize_tracked_lms(points2D[np.array(valid_pts, dtype=bool), :], frame, data)
        return pose

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
        pose = self.track_motion(slam_mapper, frame,
                                 init_pose, camera, config, data)
        return pose

    def track_LK(self, slam_mapper: SlamMapper, frame: Frame, config, camera, data):
        last_frame = slam_mapper.last_frame
        logger.debug("Tracking {} <-> {}".format(frame.im_name, last_frame.frame_id))
        print("LK Tracking: ", frame.frame_id, frame.im_name, "<->", last_frame.frame_id, last_frame.im_name)
        lk_landmarks = slam_mapper.last_lk
        if len(lk_landmarks) == 0:
            print("No landmarks, return I")
            return types.Pose()
        init_pose = slam_mapper.velocity.compose(slam_mapper.last_frame.world_pose)

        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        
        # Compute the pyramids
        # Read and convert images
        # TODO: avoid double computation of gray image and lk pyramid
        im1 = data.load_image(last_frame.im_name)
        im1_g = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
            # last_frame.lk_pyramid =\
                # cv2.buildOpticalFlowPyramid(im1_g, lk_params['winSize'], lk_params['maxLevel'])
        # if frame.lk_pyramid is None:
            # Read and convert images
        im2 = data.load_image(frame.im_name)
        im2_g = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
            # frame.lk_pyramid =\
                # cv2.buildOpticalFlowPyramid(im2_g, lk_params['winSize'], lk_params['maxLevel'])
        p0 = np.zeros([len(lk_landmarks), 3], dtype=np.float32)
        p0_3D = np.zeros([len(lk_landmarks), 3], dtype=np.float32)
        points = slam_mapper.reconstruction.points
        # valid_pts = np.ones(len(last_frame.landmarks_), dtype=np.bool)
        valid_pts = np.ones(len(lk_landmarks), dtype=np.bool)
        print("lens: ",len(last_frame.landmarks_), len(lk_landmarks))
        # lms = []
        lms = np.ones(len(lk_landmarks), dtype=np.int) * -1

        for (idx, (lm_id, p2D)) in enumerate(lk_landmarks):
            # print("idx: {}, lm_id: {}, p2D:  {}, len: {}: ".format(idx, lm_id, p2D, len(last_frame.lk_landmarks_)))
            p = points.get(lm_id)
            if p is None:
                valid_pts[idx] = False
                continue
            p0[idx, :] = p2D
            p0_3D[idx, :] = p.coordinates
            lms[idx] = int(lm_id)
            # lms.append(lm_id)
        
        cam = camera[1]
        # slam_debug.reproject_landmarks(p0_3D,None,init_pose,frame.im_name,cam,data,title="init_pose",do_show=False)
        # slam_debug.reproject_landmarks(p0_3D,None,slam_mapper.last_frame.world_pose,frame.im_name,cam,data,title="last_pose",do_show=False)
        # init_pose2 = slam_mapper.velocity.compose(slam_mapper.last_frame.world_pose).inverse()
        # slam_debug.reproject_landmarks(p0_3D,None,init_pose2,frame.im_name,cam,data,title="init_pose inv",do_show=True)
        print("p0: ", p0.shape)
        p0 = p0[valid_pts, :]
        print("p0 after: ", p0.shape)
        p0_3D = p0_3D[valid_pts, :]
        lms = lms[valid_pts]
        print(cam, camera)
        # transform landmarks to new frame
        # TODO: Remove debug stuff
        #denormalize
        p0_lk = features.\
            denormalized_image_coordinates(p0, cam.width, cam.height)
        
        camera_point = init_pose.transform_many(p0_3D)
        p1 = cam.project_many(camera_point)
        # a = np.asarray(np.arange(0,len(p0)), dtype=int)
        # slam_debug.visualize_matches_pts(p0, p1, np.column_stack((a,a)),im1, im2,False, title=last_frame.im_name+"<->"+frame.im_name+"points and reproj vel")

        # camera_point2 = slam_mapper.last_frame.world_pose.transform_many(p0_3D)
        # p12 = cam.project_many(camera_point2)
        # a = np.asarray(np.arange(0,len(p0)), dtype=int)
        # slam_debug.visualize_matches_pts(p0, p12, np.column_stack((a,a)),im1, im2,False, title=last_frame.im_name+"<->"+frame.im_name+"reproj last")

        # camera_point3 =slam_mapper.velocity.inverse().compose(slam_mapper.last_frame.world_pose).transform_many(p0_3D)
        # p13 = cam.project_many(camera_point3)
        # a = np.asarray(np.arange(0,len(p0)), dtype=int)
        # slam_debug.visualize_matches_pts(p0, p13, np.column_stack((a,a)),im1, im2,False, title=last_frame.im_name+"<->"+frame.im_name+"reproj last inv init")
        p1_init = features.\
            denormalized_image_coordinates(p1, cam.width, cam.height)
        print(p0_lk.shape, p1_init.shape)
        print(type(p0_lk), type(p1_init))
        p0_lk = np.asarray(p0_lk.reshape([-1, 1, 2]), dtype=np.float32)
        p1_init = np.asarray(p1_init.reshape([-1, 1, 2]), dtype=np.float32)
        chrono = Chronometer()                                                
        p1_lk, st, err = cv2.calcOpticalFlowPyrLK(im1_g, im2_g,
                                                  p0_lk, p1_init, **lk_params,
                                                  flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        chrono.lap('lk opt flow')
        slam_debug.avg_timings.addTimes(chrono.laps_dict)
        # Now, the points are matched
        p0_3D = p0_3D[st.flatten() == 1, :]

        # p0_3D = p0_3D.reshape([-1, 1, 3])[st == 1]
        # print(p0_3D, p0_3D_t)
        # print(np.sum(p0_3D-p0_3D_t))
        # exit()
        p1_lk = p1_init[st == 1]
        p1_lk = features.normalized_image_coordinates(p1_lk, cam.width, cam.height)
        # take std of old features
        p1_lk = np.column_stack((p1_lk, p0[st.flatten()==1, 2]))
        # p1_lk = np.column_stack((p1_lk, p0.reshape([-1,1,3])[st==1][:, 2]))
        # print(p1_lk, p1_lk_t)
        # print(p1_lk.shape, p1_lk_t.shape)
        # print("len(p0_lk): ", len(p0_lk), " len(p1): ", len(p1))
        # print("p1_lk: ", p1_lk)
        pose, valid_pts = self.bundle_tracking(p0_3D, p1_lk, init_pose,
                                               camera, config, data)
        print("visualize!")
        # slam_debug.draw_observations_in_image(p1_lk[valid_pts, :], frame.im_name, data, False)
        # slam_debug.draw_observations_in_image(p1_lk[valid_pts == False, :], frame.im_name, data, False)
        # slam_debug.draw_observations_in_image(p1_lk[valid_pts == False, :], frame.im_name, data, True)
        # a = np.asarray(np.arange(0,np.sum(valid_pts)), dtype=int)
        # slam_debug.visualize_matches_pts(p0.reshape([-1, 1, 3])[st == 1], p1_lk, np.column_stack((a,a)),im1, im2,True)

        # slam_debug.visualize_matches_pts(p0.reshape([-1, 1, 3])[st == 1][valid_pts, :], p1_lk[valid_pts, :], np.column_stack((a,a)),im1, im2,True, title=last_frame.im_name+"<->"+frame.im_name+"lk matches!")

        print("frame.landmarks_: ", len(frame.landmarks_))
        lms = lms[st.flatten()==1]
        lk_landmarks.clear()
        for idx, v in enumerate(valid_pts):
            if v:
                lk_landmarks.append((str(lms[idx]), p1_lk[idx, :]))
        
        print("len(landmarks) after: ", len(frame.landmarks_), " len(lk_landmarks): ", len(lk_landmarks))
        # print("T_init: ", T_init, " new pose for ", frame.im_name, ": ", pose.get_Rt())
        slam_debug.\
            visualize_tracked_lms(p1_lk[valid_pts, :], frame, data)
        return pose
