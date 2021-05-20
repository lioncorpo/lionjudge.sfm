from opensfm import reconstruction
from slam_types import Frame
from slam_types import Keyframe
from opensfm import types
from opensfm import features
from opensfm import csfm
import logging
logger = logging.getLogger(__name__)
import numpy as np
import networkx as nx
import slam_debug
from collections import defaultdict
import cv2
import slam_utils
import cslam
from cslam import GuidedMatcher
from cslam import SlamUtilities
from cslam import LocalMapCleaner

class SlamMapper(object):

    def __init__(self, guided_matcher, data, config_slam, camera, slam_map):
        """SlamMapper holds a local and global map
        """
        self.data = data
        self.camera = camera
        self.config = data.config
        self.config_slam = config_slam
        self.velocity = types.Pose()
        self.reconstruction = None
        # self.graph = None
        self.last_frame = None
        self.n_frames = 0
        self.n_keyframes = 0
        self.n_landmarks = 0
        self.covisibility = nx.Graph()
        self.covisibility_list = []
        self.keyframes = []
        self.c_keyframes = []
        self.local_landmarks = []
        self.c_landmarks = {}  # id, pt
        self.num_tracked_lms_thr = 15
        self.lms_ratio_thr = 0.9
        self.num_tracked_lms = 0
        self.curr_kf = None
        self.slam_map = slam_map
        self.slam_map_cleaner = cslam.LocalMapCleaner(guided_matcher, slam_map)
        self.guided_matcher = guided_matcher

    def create_reconstruction(self):
        # now we create the reconstruction
        # add only gray points
        all_kfs = self.slam_map.get_all_keyframes()
        all_landmarks = self.slam_map.get_all_landmarks()
        # reconstruction = self.reconstruction
        # add all kfs to reconstruction
        rec = types.Reconstruction()
        rec.reference = self.data.load_reference()
        rec.cameras = self.data.load_camera_models() 

        for kf in all_kfs:
            shot1 = types.Shot()
            shot1.id = kf.im_name
            shot1.camera = rec.cameras[self.camera[0]]

            T_cw = kf.get_Tcw()
            pose = types.Pose()
            pose.set_rotation_matrix(T_cw[0:3, 0:3])
            pose.translation = T_cw[0:3, 3]
            shot1.pose = pose
            shot1.metadata = reconstruction.\
                get_image_metadata(self.data, kf.im_name)
            rec.add_shot(shot1)

        for lm in all_landmarks:
            point = types.Point()
            point.id = lm.lm_id
            point.color = [255, 0, 0]
            pos_w = lm.get_pos_in_world()
            point.coordinates = pos_w.tolist()
            rec.add_point(point)

        self.reconstruction = rec

    def create_init_map(self, graph_inliers, rec_init,
                        init_frame: Frame, curr_frame: Frame,
                        init_pdc=None, other_pdc=None):
        """The graph contains the KFs/shots and landmarks.
        Edges are connections between keyframes and landmarks and
        basically "observations"
        """
        # self.last_frame = curr_frame
        # Create the keyframes
        kf1 = self.slam_map.create_new_kf(init_frame.cframe)
        kf2 = self.slam_map.create_new_kf(curr_frame.cframe)

        kf1_pose = rec_init.shots[init_frame.im_name].pose.get_Rt()
        kf1_pose = np.vstack((kf1_pose, np.array([0, 0, 0, 1])))
        kf1.set_Tcw(kf1_pose)
        # kf1.set_pose(np.linalg.inv(kf1_pose))
        kf2_pose = rec_init.shots[curr_frame.im_name].pose.get_Rt()
        kf2_pose = np.vstack((kf2_pose, np.array([0, 0, 0, 1])))
        # kf2.set_pose(np.linalg.inv(kf2_pose))
        kf2.set_Tcw(kf2_pose)
        # rec_init
        init_frame.frame_id = 0
        # Create keyframes
        init_frame.world_pose = rec_init.shots[init_frame.im_name].pose
        self.init_frame = Keyframe(init_frame, self.data, 0)
        curr_frame.frame_id = 1
        curr_frame.world_pose = rec_init.shots[curr_frame.im_name].pose
        curr_kf = Keyframe(curr_frame, self.data, 1)

        self.init_frame.ckf = kf1
        curr_kf.ckf = kf2
        # Add to data and covisibility
        self.add_keyframe(self.init_frame)
        self.add_keyframe(curr_kf)
        self.update_with_last_frame(init_frame)
        self.update_with_last_frame(curr_frame)
        for lm_id in graph_inliers[self.init_frame.im_name]:
            pos_w = rec_init.points[str(lm_id)].coordinates
            clm = self.slam_map.create_new_lm(kf1, pos_w)
            self.c_landmarks[clm.lm_id] = clm
            f1_id = graph_inliers.\
                get_edge_data(lm_id, self.init_frame.im_name)["feature_id"]
            f2_id = graph_inliers.\
                get_edge_data(lm_id, curr_kf.im_name)["feature_id"]
            # connect landmark -> kf
            clm.add_observation(kf1, f1_id)
            clm.add_observation(kf2, f2_id)
            # connect kf -> landmark in the graph
            kf1.add_landmark(clm, f1_id)
            kf2.add_landmark(clm, f2_id)
            clm.compute_descriptor()
            clm.update_normal_and_depth()
            curr_frame.cframe.add_landmark(clm, f2_id)
        print("create_init_map: len(local_landmarks): ",
              self.slam_map.get_num_landmarks())
        # Change that according to cam model
        median_depth = kf1.compute_median_depth(False)
        min_num_triangulated = 100
        # print("curr_kf.world_pose: ", curr_kf.world_pose.get_Rt)
        print("Tcw bef scale: ", kf2.get_Tcw())
        if kf2.get_num_tracked_lms(1) < min_num_triangulated and median_depth < 0:
            logger.info("Something wrong in the initialization")
        else:
            self.slam_map.scale_map(kf1, kf2, 1.0 / median_depth)
        curr_frame.world_pose = slam_utils.mat_to_pose(kf2.get_Tcw())
        print("Tcw aft scale: ", kf2.get_Tcw())
        # curr_frame.world_pose = curr_kf.world_pose
        print("Finally finished scale")

    def update_with_last_frame(self, frame: Frame):
        """Updates the last frame and the related variables in slam mapper
        """
        if self.n_frames > 0:  # we alread have frames
            self.velocity = frame.world_pose.compose(self.last_frame.world_pose.inverse())
            self.pre_last = self.last_frame
        self.n_frames += 1
        self.last_frame = frame

    def new_keyframe_is_needed(self, frame: Frame):
        num_keyfrms = len(self.c_keyframes)
        min_obs_thr = 3 if (3 <= num_keyfrms) else 2
        last_kf = self.c_keyframes[-1]
        num_reliable_lms = last_kf.get_num_tracked_lms(min_obs_thr)
        max_num_frms_ = 10  # the fps
        min_num_frms_ = 2
        # if frame.frame_id > 15 and frame.frame_id % 3:
        #     return True
        frm_id_of_last_keyfrm_ = self.curr_kf.kf_id
        print("curr_kf: ", self.curr_kf.kf_id, self.curr_kf.frame_id)
        print("frame.frame_id: ", frame.frame_id, frm_id_of_last_keyfrm_)
        # frame.id
        # ## mapping: Whether is processing
        # #const bool mapper_is_idle = mapper_->get_keyframe_acceptability();
        # Condition A1: Add keyframes if max_num_frames_ or more have passed

        # since the last keyframe insertion
        cond_a1 = (frm_id_of_last_keyfrm_ + max_num_frms_ <= frame.frame_id)
        # Condition A2: Add keyframe if min_num_frames_ or more has passed
        # and mapping module is in standby state
        cond_a2 = (frm_id_of_last_keyfrm_ + min_num_frms_ <= frame.frame_id)
        # cond_a2 = False
        # Condition A3: Add a key frame if the viewpoint has moved from the
        # previous key frame
        cond_a3 = self.num_tracked_lms < (num_reliable_lms * 0.25)

        print("self.num_tracked_lms_thr {} self.num_tracked_lms {}\n \
               num_reliable_lms {} * self.lms_ratio_th={}".
              format(self.num_tracked_lms_thr, self.num_tracked_lms,
                     num_reliable_lms, num_reliable_lms * self.lms_ratio_thr))
        # Condition B: (Requirement for adding keyframes)
        # Add a keyframe if 3D points are observed above the threshold and
        # the percentage of 3D points is below a certain percentage
        cond_b = (self.num_tracked_lms_thr <= self.num_tracked_lms) and \
                 (self.num_tracked_lms < num_reliable_lms * self.lms_ratio_thr)

        print("cond_a1: {}, cond_a2: {}, cond_a3: {}, cond_b: {}"
              .format(cond_a1, cond_a2, cond_a3, cond_b))

        # Do not add if B is not satisfied
        if not cond_b:
            print("not cond_b -> no kf")
            return False
        
        # Do not add if none of A is satisfied
        if not cond_a1 and not cond_a2 and not cond_a3:
            print("not cond_a1 and not cond_a2 and not cond_a3 -> no kf")
            return False
        print("NEW KF", frame.im_name)
        return True

    # == mapping_with_new_keyframe
    def insert_new_keyframe(self, frame: Frame):
        # Create new Keyframe
        new_kf = Keyframe(frame, self.data, self.n_keyframes)
        new_kf.ckf = self.slam_map.create_new_kf(frame.cframe)
        kf1_pose = np.vstack((new_kf.world_pose.get_Rt(), np.array([0, 0, 0, 1])))
        new_kf.ckf.set_Tcw(kf1_pose)
        self.add_keyframe(new_kf)  
        self.slam_map_cleaner.update_lms_after_kf_insert(new_kf.ckf)
        self.slam_map_cleaner.remove_redundant_lms(new_kf.kf_id)
        # min_d, max_d = GuidedMatcher.compute_min_max_depth(frame.cframe)
        # of_vec = GuidedMatcher.compute_optical_flow(frame.cframe)
        # print("of_vec", of_vec)
        print("create_new_landmarks_before")
        chrono = reconstruction.Chronometer()
        self.create_new_landmarks()
        chrono.lap("create_landmarks")
        print("create_new_landmarks_after")
        self.update_new_keyframe()
        chrono.lap("update_keyframe")
        if self.n_keyframes % self.config_slam["run_local_ba_every_nth"] == 0:
            self.local_bundle_adjustment()
        chrono.lap("local_bundle_adjustment")
        slam_debug.avg_timings.addTimes(chrono.laps_dict)

        if self.n_keyframes % 50 == 0:
            chrono.start()
            self.create_reconstruction()
            self.save_reconstruction(frame.im_name + "aft")
            chrono.lap("create+save rec")
            slam_debug.avg_timings.addTimes(chrono.laps_dict)
        chrono.start()
        # n_kf_removed = self.remove_redundant_kfs()
        n_kf_removed = self.slam_map_cleaner.remove_redundant_kfs(new_kf.ckf, self.c_keyframes[0].kf_id)
        print("Removed {} keyframes ".format(n_kf_removed))
        if (n_kf_removed > 0):
            print("Finally removed frames")
        chrono.lap("remove_redundant_kfs")
        slam_debug.avg_timings.addTimes(chrono.laps_dict)

    def update_new_keyframe(self):
        """ update new keyframe
        detect and resolve the duplication of the landmarks observed in the current frame
        """      
        # again, check the last 10 frames
        # fuse_kfs = self.c_keyframes[-5:-1]
        fuse_kfs = cslam.SlamUtilities.get_second_order_covisibilities_for_kf(self.curr_kf.ckf, 20, 5)
        print("update_new_keyframe fuse")
        for kf in fuse_kfs:
            print("get_second_order_covisibilities_for_kf kf: ", kf.im_name)
        # slam_debug.visualize_tracked_lms(self.curr_kf.ckf.get_valid_kpts(), frame, data)
        # im = self.data.load_image(self.curr_kf.ckf.im_name)
        # slam_debug.disable_debug = False
        # slam_debug.draw_obs_in_image_no_norm(self.curr_kf.ckf.get_valid_keypts(), im, title="bef fuse", do_show=False)
        self.slam_map_cleaner.\
            fuse_landmark_duplication(self.curr_kf.ckf, list(fuse_kfs))
        # slam_debug.draw_obs_in_image_no_norm(self.curr_kf.ckf.get_valid_keypts(), im, title="aft fuse", do_show=True)
        # slam_debug.disable_debug = False
        print("update_new_keyframe fuse done")
        cslam.SlamUtilities.update_new_keyframe(self.curr_kf.ckf)
        self.curr_kf.ckf.get_graph_node().update_connections()
        print("update_new_keyframe done")

    def local_bundle_adjustment(self):
        """ TODO: Build optimization problem directly from C++"""
        if self.n_keyframes <= 2:
            return

        ba = csfm.BundleAdjuster()
        # Find "earliest" KF seen by the current map! 
        # Add new landmarks to optimize
        # local_keyframes = self.c_keyframes[-n_kfs_optimize: -1]

        # (1), find all the kfs that see landmarks in the current frame and let's
        #       call them local keyframes
        # (2) find all the landmarks seen in local keyframes
        # (3) find all the keyframes containing the landmarks but set the ones
        #     not in local keyframes constant 
        # correct local keyframes of the current keyframe
        kf_added = {}
        cam = self.camera[1]
        reconstruction._add_camera_to_bundle(ba, cam, cam, constant=True)
        cam_id = str(self.camera[0])
        local_kfs_idx = self.curr_kf.ckf.compute_local_keyframes()
        local_kfs = []
        kfs_dict_constant = {}
        # Get the local keyframes
        for kf_id in local_kfs_idx:
            kf = self.c_keyframes[kf_id]
            local_kfs.append(kf)
            # add them directly to BA problem
            T_cw = kf.get_Tcw()
            R_cw = cv2.Rodrigues(T_cw[0:3, 0:3])[0]
            t_cw = T_cw[0:3, 3]
            ba.add_shot(str(kf_id), cam_id, R_cw, t_cw, kf_id == 0)
            kf_added[kf_id] = True
            kfs_dict_constant[kf_id] = True if kf_id == 0 else False

        # Get the landmarks from the keyframes
        # From the local keyframes, get the landmarks
        # and the non-local keyframes
        lm_kf_added = set()
        lm_added = set()
        for kf_id in local_kfs_idx:
            kf = self.c_keyframes[kf_id]
            lms = kf.get_valid_lms()
            points2D, _, _ = features.\
                normalize_features(kf.get_valid_keypts(), None, None,
                                   cam.width, cam.height)

            for (lm, pt2D) in zip(lms, points2D):
                lm_id = lm.lm_id
                ba.add_point(str(lm_id), lm.get_pos_in_world(), False)
                ba.add_point_projection_observation(str(kf_id), str(lm_id),
                                                    pt2D[0], pt2D[1], pt2D[2])
                lm_kf_added.add((lm_id, kf_id))
                lm_added.add(lm)


        # test something
        kf = self.c_keyframes[-1]
        lms = kf.get_valid_lms()
        points2D = kf.get_valid_keypts()
        points3D = np.zeros((len(lms), 3), dtype=np.float)
        for idx, lm in enumerate(lms):
            points3D[idx, :] = lm.get_pos_in_world()
        slam_debug.disable_debug = True
        slam_debug.reproject_landmarks(points3D, points2D, 
            slam_utils.mat_to_pose(kf.get_Tcw()), self.data.load_image(kf.im_name), self.camera[1], title="repro", do_show=True, obs_normalized=False)
        slam_debug.disable_debug = True
        # End test something

        # Go through the added landmarks and add the keyframes
        # that are not in local keyframes
        # Now, get all the keyframes that are not in local keyframes
        # from the landmarks and fix their poses
        for lm in lm_added:
            kf_idx_list = lm.get_observations()
            for kf, idx in kf_idx_list.items():
                kf_id = kf.kf_id
                lm_id = lm.lm_id
                if (kf_id, lm_id) in lm_kf_added:
                    continue
                lm_kf_added.add((kf_id, lm_id))
                if kf_added.get(kf_id) is None:
                    # add the kf
                    T_cw = kf.get_Tcw()
                    R_cw = cv2.Rodrigues(T_cw[0:3, 0:3])[0]
                    t_cw = T_cw[0:3, 3]
                    ba.add_shot(str(kf_id), cam_id, R_cw, t_cw, True)
                    kf_added[kf_id] = True
                    kfs_dict_constant[kf_id] = True
                # add reprojections
                pt = kf.get_obs_by_idx(idx)
                # pt2D, _, _ = features.normalize_features(pt, None, None, cam.width, cam.height)
                pt2D, _, _ = features.normalize_features(pt.reshape((1, 3)), None, None, cam.width, cam.height)
                pt2D = pt2D.reshape((3, 1))
                ba.add_point_projection_observation(str(kf_id), str(lm_id), pt2D[0], pt2D[1], pt2D[2])

        config = self.config
        # Assume observations N x 3 (x,y,s)
        ba.add_absolute_up_vector(str(local_kfs_idx[0]), [0, 0, -1], 1e-3)
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
        ba.run()

        # TODO: check outliers!
        # print("ba.full_report(): ", ba.full_report())
        # Update landmarks
        lms = self.curr_kf.ckf.get_valid_lms()
        for lm in lms:
            pos_w = ba.get_point(str(lm.lm_id)).p
            lm.set_pos_in_world(pos_w)
            # lm

        # DEBUG
        points3D = np.zeros((len(lms), 3))
        for idx, lm in enumerate(lms):
            lm_idx = str(lm.lm_id)
            point = ba.get_point(lm_idx)
            # print("point.reprojection_errors: ", point.reprojection_errors)
            pos_w = point.p
            n_th = 0
            th = 0.006
            for (k, v) in point.reprojection_errors.items():
                if np.linalg.norm(v) > th:
                    # print("remove lm_id: ", lm_idx, " kf_id", k)
                    # remove outlier observations
                    # self.graph.remove_edge(lm_id, k)
                    # k -> kf_id
                    n_th += 1
            points3D[idx, :] = pos_w
        print("Found: ", n_th, " outliers!")
        slam_debug.disable_debug = True
        slam_debug.reproject_landmarks(points3D, None, self.curr_kf.world_pose, self.data.load_image(self.curr_kf.im_name), self.camera[1], do_show=False, title="bef")
        shot = ba.get_shot(str(self.curr_kf.kf_id))
        pose = types.Pose(shot.r, shot.t)
        slam_debug.reproject_landmarks(points3D, None, pose, self.data.load_image(self.curr_kf.im_name), self.camera[1], do_show=True, title="aft")
        slam_debug.disable_debug = True
        # DEBUG END

        # Update keyframes
        for kf_id, constant in kfs_dict_constant.items():
            if not constant:
                kf = self.c_keyframes[kf_id]
                shot = ba.get_shot(str(kf.kf_id))
                pose = types.Pose(shot.r, shot.t)
                kf.set_Tcw(np.vstack((pose.get_Rt(), np.array([0, 0, 0, 1]))))

    def create_new_landmarks(self):
        """Creates a new landmarks with using the newly added KF
        """
        new_kf = self.c_keyframes[-1]
        new_im = self.data.load_image(new_kf.im_name)
        new_cam_center = new_kf.get_cam_center()
        new_Tcw = new_kf.get_Tcw()
        new_R = new_Tcw[0:3, 0:3]
        new_t = new_Tcw[0:3, 3]
        # Again, just take the last 10 frames
        # but not the current one!
        num_covisibilities = 10
        # TODO: replace "local" keyframes by that
        cov_kfs = new_kf.get_graph_node().get_top_n_covisibilities(2 * num_covisibilities)
        local_keyframes = self.c_keyframes[-5:-1]
        print("local_kf: ", len(local_keyframes))
        # TODO! check new_kf pose
        # for (old_kf, old_kf_py) in zip(local_keyframes, py_kfs):
        n_baseline_reject = 0
        chrono = reconstruction.Chronometer()
        new_med_depth = new_kf.compute_median_depth(True)
        min_d, max_d = GuidedMatcher.compute_min_max_depth(new_kf)
        # min_d *= 0.5
        # max_d *= 2
        for old_kf in local_keyframes:
            old_cam_center = old_kf.get_cam_center()
            baseline_vec = old_cam_center - new_cam_center
            baseline_dist = np.linalg.norm(baseline_vec)
            median_depth_in_old = old_kf.compute_median_depth(True)
            if baseline_dist < 0.02 * median_depth_in_old:
                n_baseline_reject += 1
                continue

            # Compute essential matrix!
            old_Tcw = old_kf.get_Tcw()
            old_R = old_Tcw[0:3, 0:3]
            old_t = old_Tcw[0:3, 3]
            
            chrono.start()
            E_old_to_new = self.guided_matcher.create_E_21(new_R, new_t, old_R, old_t)
            chrono.lap("compute E")
            # matches_old = self.guided_matcher.match_for_triangulation(new_kf, old_kf, E_old_to_new)
            # chrono.lap("match_for_triangulation")
            # matches = self.guided_matcher.match_for_triangulation_with_depth(new_kf, old_kf, E_old_to_new, new_med_depth)
            # matches1 = self.guided_matcher.match_for_triangulation_epipolar(new_kf, old_kf, E_old_to_new, min_d, max_d, True, 10)
            # chrono.lap("match_for_triangulation_depth")
            matches = self.guided_matcher.match_for_triangulation_epipolar(new_kf, old_kf, E_old_to_new, min_d, max_d, False, 10)
            chrono.lap("match_for_triangulation_line_10")
            # matches_5 = self.guided_matcher.match_for_triangulation_epipolar(new_kf, old_kf, E_old_to_new, min_d, max_d, False, 5)
            # matches_5_2 = self.guided_matcher.match_for_triangulation_epipolar(new_kf, old_kf, E_old_to_new, min_d, max_d, False, 5)
            # chrono.lap("match_for_triangulation_line_5")
            # matches_20 = self.guided_matcher.match_for_triangulation_epipolar(new_kf, old_kf, E_old_to_new, min_d, max_d, False, 20)
            # chrono.lap("match_for_triangulation_line_20")
            # print("Matching: old", chrono.lap_time("match_for_triangulation"), "vs depth", 
            #       chrono.lap_time("match_for_triangulation_depth"), 
            #       " vs match_for_triangulation_line_10 ", chrono.lap_time("match_for_triangulation_line_10"),
            #       " vs match_for_triangulation_line_20 ", chrono.lap_time("match_for_triangulation_line_20"),
            #       " vs match_for_triangulation_line_5 ", chrono.lap_time("match_for_triangulation_line_5"))
            # print("Matches old: ", len(matches_old), " vs depth ", len(matches1), "vs epi10", len(matches), "vs epi5", len(matches_5), " vs epi20", len(matches_20))
            old_im = self.data.load_image(old_kf.im_name)
            # slam_debug.disable_debug = False
            # slam_debug.visualize_matches_pts(
            #     new_kf.getKptsPy(), old_kf.getKptsPy(), np.array(matches_old),
            #     new_im, old_im, do_show=False, is_normalized=False,
            #     title=old_kf.im_name+"old")
            # slam_debug.visualize_matches_pts(
            #     new_kf.getKptsPy(), old_kf.getKptsPy(), np.array(matches1),
            #     new_im, old_im, do_show=False, is_normalized=False,
            #     title=old_kf.im_name+"depth")
            # slam_debug.visualize_matches_pts(
            #     new_kf.getKptsPy(), old_kf.getKptsPy(), np.array(matches),
            #     new_im, old_im, do_show=False, is_normalized=False,
            #     title=old_kf.im_name+"epi10")    
            # slam_debug.visualize_matches_pts(
            #     new_kf.getKptsPy(), old_kf.getKptsPy(), np.array(matches_5),
            #     new_im, old_im, do_show=False, is_normalized=False,
            #     title=old_kf.im_name+"epi5")
            
            # T_1_2 = old_kf.get_Tcw().dot(new_kf.get_Twc())
            # K = self.camera[1].get_K_in_pixel_coordinates()
            # # TODO: Iterate through depths, test undistorted points!
            # # Questions: Why is the  low depth farther away from the camera??
            
            # slam_debug.visualize_epipolar_line(old_kf.getKptsUndist(), new_kf.getKptsUndist(),
            #                                    old_im, new_im, T_1_2, K, min_d, max_d)
            # slam_debug.visualize_matches_pts(
            #     new_kf.getKptsPy(), old_kf.getKptsPy(), np.array(matches_20),
            #     new_im, old_im, do_show=True, is_normalized=False,
            #     title=old_kf.im_name+"epi20")

            # slam_debug.disable_debug = True
            
            self.triangulate_from_two_kfs(new_kf, old_kf, matches)
            chrono.lap("triangulate_from_two_kfs")
            slam_debug.avg_timings.addTimes(chrono.laps_dict)
        print("n_baseline_reject: ", n_baseline_reject)

    def triangulate_from_two_kfs(self, new_kf, old_kf, matches):
        # TODO: try without tracks graph
        frame1 = new_kf.im_name
        frame2 = old_kf.im_name
        # create the graph
        tracks_graph = nx.Graph()
        tracks_graph.add_node(str(frame1), bipartite=0)
        tracks_graph.add_node(str(frame2), bipartite=0)
        f_processed = defaultdict(int)
        p1, _, _ = features.\
            normalize_features(new_kf.getKptsPy(), None, None,
                               self.camera[1].width, self.camera[1].height)
        p2, _, _ = features.\
            normalize_features(old_kf.getKptsPy(), None, None,
                               self.camera[1].width, self.camera[1].height)
        # c1 = new_kf.colors
        # c2 = old_kf.colors

        for (track_id, (f1_id, f2_id)) in enumerate(matches):
            # this checks whether the current kf was matched
            # to one of the landmarks.
            # if f2 is already in a lm
            f_processed[f1_id] += 1
            if f_processed[f1_id] > 1:
                print("double add!!")
                exit()
            x, y, s = p2[f2_id, 0:3]
            r, g, b = [0 , 0, 0] #c2[f2_id, :]
            tracks_graph.add_node(str(track_id), bipartite=1)
            tracks_graph.add_edge(str(frame2),
                                  str(track_id),
                                  feature=(float(x), float(y)),
                                  feature_scale=float(s),
                                  feature_id=int(f2_id),
                                  feature_color=(float(r), float(g), float(b)))

            x, y, s = p1[f1_id, 0:3]
            r, g, b = [0, 0, 0] #c1[f1_id, :]
            tracks_graph.add_edge(str(frame1),
                                  str(track_id),
                                  feature=(float(x), float(y)),
                                  feature_scale=float(s),
                                  feature_id=int(f1_id),
                                  feature_color=(float(r), float(g), float(b)))
        # chrono.lap("track_graph")
        cameras = self.data.load_camera_models()
        camera = next(iter(cameras.values()))
        rec_tri = types.Reconstruction()
        rec_tri.reference = self.data.load_reference()
        rec_tri.cameras = cameras
        pose1 = slam_utils.mat_to_pose(new_kf.get_Tcw())
        shot1 = types.Shot()
        shot1.id = frame1
        shot1.camera = camera
        shot1.pose = pose1
        shot1.metadata = reconstruction.get_image_metadata(self.data, frame1)
        rec_tri.add_shot(shot1)

        pose2 = slam_utils.mat_to_pose(old_kf.get_Tcw())
        shot2 = types.Shot()
        shot2.id = frame2
        shot2.camera = camera
        shot2.pose = pose2
        shot2.metadata = reconstruction.get_image_metadata(self.data, frame2)
        rec_tri.add_shot(shot2)

        graph_inliers = nx.Graph()
        # chrono.lap("ba setup")
        np_before = len(rec_tri.points)
        reconstruction.triangulate_shot_features(tracks_graph, graph_inliers,
                                                 rec_tri, frame1,
                                                 self.data.config)
        np_after = len(rec_tri.points)
        print("Successfully triangulated {} out of {} points.".
              format(np_after, np_before))
        # chrono.lap("triangulateion")
        # edges1 = graph_inliers.edges(frame1)
        points = rec_tri.points
        points3D = np.zeros((len(points), 3))
        for idx, pt3D in enumerate(points.values()):
            points3D[idx, :] = pt3D.coordinates

        slam_debug.reproject_landmarks(points3D, None, slam_utils.mat_to_pose(new_kf.get_Tcw()), self.data.load_image(new_kf.im_name), self.camera[1], do_show=False)
        slam_debug.reproject_landmarks(points3D, None, slam_utils.mat_to_pose(old_kf.get_Tcw()), self.data.load_image(old_kf.im_name), self.camera[1], do_show=True)
        kf1 = new_kf
        kf2 = old_kf
        # Add to graph -> or better just create clm
        for _, gi_lm_id in graph_inliers.edges(frame1):
            # TODO: Write something like create_landmark
            pos_w = rec_tri.points[gi_lm_id].coordinates
            clm = self.slam_map.create_new_lm(kf2, pos_w)
            self.c_landmarks[clm.lm_id] = clm
            e1 = graph_inliers.get_edge_data(frame1, gi_lm_id)
            e2 = graph_inliers.get_edge_data(frame2, gi_lm_id)
            f1_id = e1['feature_id']
            f2_id = e2['feature_id']
            # connect landmark -> kf
            clm.add_observation(kf1, f1_id)
            clm.add_observation(kf2, f2_id)
            kf1.add_landmark(clm, f1_id)
            kf2.add_landmark(clm, f2_id)
            clm.compute_descriptor()
            clm.update_normal_and_depth()
            self.last_frame.cframe.add_landmark(clm, f1_id)
            self.slam_map_cleaner.add_landmark(clm)
            # TODO: check if this the correct frame

            # We also have to add the points to the reconstruction
            # point = types.Point()
            # point.id = str(lm_id)
            # point.coordinates = rec_tri.points[gi_lm_id].coordinates
            # self.reconstruction.add_point(point)

    def add_keyframe(self, kf: Keyframe):
        """Adds a keyframe to the map graph
        and the covisibility graph
        """
        logger.debug("Adding new keyframe # {}, {}".format(kf.kf_id, kf.im_name))

        # and to covisibilty graph
        self.covisibility.add_node(str(kf.im_name))
        self.covisibility_list.append(str(kf.im_name))
        self.n_keyframes += 1
        self.keyframes.append(kf)
        self.c_keyframes.append(kf.ckf)
        self.curr_kf = kf

    def paint_reconstruction(self):
        if self.reconstruction is not None and self.graph is not None:
            reconstruction.paint_reconstruction(self.data, self.graph,
                                                self.reconstruction)

    def save_reconstruction(self, name: str):
        if self.reconstruction is not None:
            logger.debug("Saving reconstruction with {} points and {} frames".
                         format(len(self.reconstruction.points),
                                len(self.reconstruction.shots)))
            self.data.save_reconstruction([self.reconstruction],
                                     'reconstruction' + name + '.json')
