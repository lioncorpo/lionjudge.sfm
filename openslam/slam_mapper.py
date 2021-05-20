from opensfm import types
from opensfm import reconstruction
from opensfm import feature_loader
from opensfm import csfm
from opensfm.reconstruction import Chronometer
from slam_types import Frame
from slam_types import Keyframe
from slam_types import Landmark
import slam_debug
import slam_utils
# from slam_tracker import SlamTracker
# from slam_matcher import SlamMatcher
import slam_matcher
from collections import defaultdict
import networkx as nx
import logging
import numpy as np
logger = logging.getLogger(__name__)
from itertools import compress


class SlamMapper(object):

    def __init__(self, data, config, config_slam, camera):
        """SlamMapper holds a local and global map
        """
        self.data = data
        self.camera = camera
        self.last_lk = []
        self.velocity = types.Pose()  # relative motion between curr frame and the one before
        self.last_frame = Frame("dummy", -1)
        # Threshold of the ratio of the number of 3D points observed in the
        # current frame to the number of 3D points observed in the latest KF
        self.num_tracked_lms_thr = 15
        self.num_tracked_lms = 0
        self.lms_ratio_thr = 0.9
        self.feature_ids_last_frame = None
        self.n_tracks = 0
        self.graph = nx.Graph()
        self.reconstruction = []
        self.n_landmarks = 0  # == unique lm id
        self.n_keyframes = 0  # == unique kf id
        self.n_frames = 0     # == unique frame id
        self.curr_kf = None
        # dict because frames can be deleted
        self.keyframes = []     # holds the id, Frame()
        self.local_keyframes = []  # keyframes in the local map
        self.local_landmarks = []  # landmarks in the local keyframes
        self.covisibility = nx.Graph()
        self.covisibility_list = []
        self.fresh_landmarks = []
        self.current_lm_i = 0
        self.config_slam = config_slam



    def create_init_map(self, graph_inliers, rec_init,
                        init_frame: Frame, curr_frame: Frame,
                        init_pdc=None, other_pdc=None):
        """The graph contains the KFs/shots and landmarks.
        Edges are connections between keyframes and landmarks and
        basically "observations"
        """
        # Store the initial graph and reconstruction
        self.graph = graph_inliers
        self.reconstruction = rec_init
        init_frame.frame_id = 0
        # Create keyframes
        self.init_frame = Keyframe(init_frame, self.data, 0, init_pdc)
        self.init_frame.world_pose = \
            rec_init.shots[init_frame.im_name].pose
        curr_frame.frame_id = 1
        curr_kf = Keyframe(curr_frame, self.data, 1, other_pdc)
        curr_kf.world_pose = rec_init.shots[curr_frame.im_name].pose

        # Add to data and covisibility
        self.add_keyframe(self.init_frame)
        self.add_keyframe(curr_kf)
        self.n_frames = 2
        # self.velocity = curr_kf.world_pose.compose(self.init_frame.world_pose.inverse())
        max_lm = 0  # find the highest lm id

        # debug
        p0_3D = np.zeros([len(self.graph[self.init_frame.im_name]), 3], dtype=np.float32)
        p0 = np.zeros([len(self.graph[self.init_frame.im_name]), 2], dtype=np.float32)
        for idx, lm_id in enumerate(self.graph[self.init_frame.im_name]):
            p0_3D[idx, :] = rec_init.points[str(lm_id)].coordinates
            p0[idx, :] = self.graph.get_edge_data(init_frame.im_name, str(lm_id))['feature']
        im1 = self.data.load_image(self.init_frame.im_name)
        im2 = self.data.load_image(curr_kf.im_name)
        # project landmarks into kf1
        # TODO debug remove
        cam = self.camera[1]
        camera_point = self.init_frame.world_pose.transform_many(p0_3D)
        p1 = cam.project_many(camera_point)
        slam_debug.disable_debug = False

        a = np.asarray(np.arange(0, len(p0)), dtype=int)
        slam_debug.visualize_matches_pts(p0, p1, np.column_stack((a, a)), im1, im1, False, title="to kf1")
        # project landmarks into kf2
        camera_point2 = curr_kf.world_pose.transform_many(p0_3D)
        p12 = cam.project_many(camera_point2)
        a = np.asarray(np.arange(0, len(p0)), dtype=int)
        slam_debug.visualize_matches_pts(p0, p12, np.column_stack((a, a)), im1, im2, False, title="to kf2")
        # project landmarks into coordinate system of kf 1 and then to kf2
        camera_point3 = self.velocity.compose(self.init_frame.world_pose).transform_many(p0_3D)
        p13 = cam.project_many(camera_point3)
        a = np.asarray(np.arange(0, len(p0)), dtype=int)
        slam_debug.visualize_matches_pts(p0, p13, np.column_stack((a, a)), im1, im2, True, title="to kf1 and then 2")
        slam_debug.disable_debug = True
        # debug end
        n_lm_added = 0
        # Add landmark objects to nodes
        for lm_id in self.graph[self.init_frame.im_name]:
            lm = Landmark(int(lm_id))
            lm.num_observable = 2
            lm.num_observed = 2
            lm.first_kf_id = self.init_frame.kf_id
            self.graph.add_node(lm_id, data=lm)

            int_id = int(lm_id)
            if int_id > max_lm:
                max_lm = int_id
            lm.compute_descriptor(curr_kf, self.graph)
            pos_w = rec_init.points[str(lm_id)].coordinates
            lm.update_normal_and_depth(pos_w, self.graph)
            self.local_landmarks.append(lm_id)
            n_lm_added += 1

        print("create_init_map: len(local_landmarks): ", len(self.local_landmarks), n_lm_added)
        self.current_lm_id = max_lm+1

        # also copy them to current kf
        curr_kf.landmarks_ = self.local_landmarks.copy()
        self.init_frame.landmarks_ = self.local_landmarks.copy()
        curr_frame.landmarks_ = self.local_landmarks.copy()
        # obs = []
        self.last_lk.clear()
        
        # go through the graph
        for lm_id in graph_inliers[init_frame.im_name]:
            # get the feature ids
            f1 = self.graph.get_edge_data(init_frame.im_name, lm_id)['feature_id']
            f2 = self.graph.get_edge_data(curr_frame.im_name, lm_id)['feature_id']
            self.init_frame.matched_lms[f1] = lm_id
            curr_kf.matched_lms[f2] = lm_id
            # print("create: ", init_frame.im_name, ":" ,self.graph.get_edge_data(init_frame.im_name, lm_id),
            #       curr_frame.im_name, ": ", self.graph.get_edge_data(curr_frame.im_name, lm_id))
            # for the LK tracker
            f = self.graph.get_edge_data(curr_frame.im_name, lm_id)
            self.last_lk.append((lm_id, np.hstack((f['feature'], f['feature_scale']))))

        # copy local landmarks to last_frame
        self.last_frame.landmarks_ = curr_kf.landmarks_.copy()
        # self.last_frame.lk_landmarks_ = curr_frame.lk_landmarks_.copy()
        self.last_frame.im_name = curr_kf.im_name
        self.last_frame.world_pose = curr_kf.world_pose  # init pose = I, thus it is correct
        self.n_landmarks = len(self.last_lk)
        # Set velocity to identity because we do not know the distance between init and current kf, e.g. frame 0 <-> 5
        # is 5 x the distance between consecutive frames
        self.velocity = types.Pose() 
        print("create_init_map with landmarks: ", len(curr_kf.landmarks_),
              len(self.last_frame.landmarks_), len(self.local_landmarks))
        self.update_local_map(curr_frame)
        print("create_init_map up with landmarks: ", len(curr_kf.landmarks_),
              len(self.last_frame.landmarks_), len(self.local_landmarks))
        self.mapping_with_new_keyframe(self.init_frame)
        print("create_init_map map new with landmarks: ", len(curr_kf.landmarks_),
              len(self.last_frame.landmarks_), len(self.local_landmarks))
        self.mapping_with_new_keyframe(curr_kf)
        print("after create_init_map with landmarks: ", len(curr_kf.landmarks_),
              len(self.last_frame.landmarks_), len(self.local_landmarks))

    def add_keyframe(self, kf: Keyframe):
        """Adds a keyframe to the map graph
        and the covisibility graph
        """
        # add kf object to existing graph node
        self.graph.add_node(str(kf.im_name), bipartite=0, data=kf)
        self.covisibility.add_node(str(kf.im_name))
        self.covisibility_list.append(str(kf.im_name))
        self.n_keyframes += 1
        shot1 = types.Shot()
        shot1.id = kf.im_name
        shot1.camera = self.camera[1]
        print("kf.im_name: ", kf.im_name, "camera: ", self.camera)
        shot1.pose = kf.world_pose
        shot1.metadata = reconstruction.\
            get_image_metadata(self.data, kf.im_name)
        self.reconstruction.add_shot(shot1)
        self.keyframes.append(kf.im_name)
        self.local_keyframes.append(kf.im_name)

    def add_landmark(self, lm: Landmark):
        """Add landmark to graph"""
        self.graph.add_node(str(lm.lm_id), bipartite=1, data=lm)

    def fuse_duplicated_landmarks(self):
        # return
        print("self.local_keyframes", self.local_keyframes)
        duplicates = 0
        n_original = 0
        for kf_id in self.local_keyframes:
            # read all the landmarks attached to this keyframe
            landmarks = self.graph[kf_id]
            feature_ids = {}
            for lm_id in landmarks:
                edge = self.graph.get_edge_data(kf_id, lm_id)
                feature_id = edge['feature_id']
                elem = feature_ids.get(feature_id)
                if elem is not None:
                    duplicates += 1
                    print("Found duplicate at ", elem, " for ", lm_id)
                    print("elem: ", elem, self.graph[elem])
                    print("lm_id: ", lm_id, self.graph[lm_id])
                    print("edge_elem: ", self.graph.get_edge_data(kf_id, elem))
                    print("edge_lm_id: ", edge)
                    exit()
                else:
                    feature_ids[feature_id] = lm_id
                    n_original += 1
        # create a dict with feature ids
        print("duplicates found: ", duplicates, n_original - duplicates)

        # OpenVSlam style
        # reproject the landmarks observed in the current keyframe to each of the targets, and acquire
        # - additional matches
        # - duplication of matches
        # then, add matches and solve duplication

        # reproject the landmarks observed in each of the targets to each of the current frame, and acquire
        # - additional matches
        # - duplication of matches
        # then, add matches and solve duplication
        pass

    def update_with_last_frame(self, frame: Frame):
        """Updates the last frame and the related variables in slam mapper
        """
        self.n_frames += 1
        self.velocity = frame.world_pose.compose(self.last_frame.world_pose.inverse())

        # debug
        p0_3D = np.zeros([len(self.graph[self.init_frame.im_name]), 3], dtype=np.float32)
        p0 = np.zeros([len(self.graph[self.init_frame.im_name]), 2], dtype=np.float32)
        for idx, lm_id in enumerate(self.graph[self.init_frame.im_name]):
            p0_3D[idx, :] = self.reconstruction.points[str(lm_id)].coordinates
            p0[idx, :] = self.graph.get_edge_data(self.init_frame.im_name, str(lm_id))['feature']
        im1, im2 = self.data.load_image(self.init_frame.im_name), self.data.load_image(frame.im_name)
        im3 = self.data.load_image(self.last_frame.im_name)
        # project landmarks into kf1
        cam  = self.camera[1]
        # camera_point = self.init_frame.world_pose.transform_many(p0_3D)
        camera_point = frame.world_pose.transform_many(p0_3D)
        p1 = cam.project_many(camera_point)
        a = np.asarray(np.arange(0,len(p0)), dtype=int)
        slam_debug.visualize_matches_pts(p0, p1, np.column_stack((a, a)), im1, im2, False, title="to frame"+frame.im_name)
        # project landmarks into kf2
        camera_point2 =self.last_frame.world_pose.transform_many(p0_3D)
        p12 = cam.project_many(camera_point2)
        a = np.asarray(np.arange(0,len(p0)), dtype=int)
        slam_debug.visualize_matches_pts(p0, p12, np.column_stack((a, a)), im1, im3, False, title="to last frame"+self.last_frame.im_name)
        # project landmarks into coordinate system of kf 1 and then to kf2
        camera_point3 = self.velocity.compose(self.last_frame.world_pose).transform_many(p0_3D)
        p13 = cam.project_many(camera_point3)
        a = np.asarray(np.arange(0,len(p0)), dtype=int)
        slam_debug.visualize_matches_pts(p0, p13, np.column_stack((a, a)), im1, im2, True, title="to last frame and then frame")
        # debug end

        self.last_frame = frame
        self.num_tracked_lms = len(self.last_lk) #len(frame.lk_landmarks_)
        print("self.num_tracked_lms {} vs lms in last kf {}, ratio {}".format(self.num_tracked_lms, len(self.graph[self.keyframes[-1]]), self.num_tracked_lms/len(self.graph[self.keyframes[-1]])))
        # now toogle all the landmarks as observed
        # for lm_id, _ in frame.lk_landmarks_:
        #     # print("Trying to fetch: ", lm_id)
        #     lm = self.graph.node[lm_id]['data']
        #     lm.num_observable += 1
        print("Update with last frame {}, {}".format(self.n_frames, len(frame.landmarks_)))

    def set_last_frame(self, frame: Frame):
        """Sets the last frame

        Arguments:
            frame: of Frame
        """
        self.n_frames += 1
        print("set_last_frame 1: ", len(frame.landmarks_))
        self.last_frame = frame
        print("set_last_frame: ", frame.im_name, self.last_frame.im_name,
              len(frame.landmarks_), len(self.last_frame.landmarks_))
        print("set_last_frame: ", frame.frame_id, "/", self.n_frames)

    def paint_reconstruction(self, data):
        reconstruction.paint_reconstruction(data, self.graph,
                                            self.reconstruction)

    def save_reconstruction(self, data, name: str):
        print("len(reconstruction): ", len(self.reconstruction.points))
        print("len(shots)", len(self.reconstruction.shots))
        data.save_reconstruction([self.reconstruction],
                                 'reconstruction'+name+'.json')

    # def clean_landmarks(self):
    #     return True

    def update_local_keyframes_lk(self, frame: Frame):
        """Count number of lm shared between current frame and neighbour KFs
        (count obs.). For each keyframe, we keep count of how many lms it
        shares with the current one.
        """
        print("update_local_keyframes")
        kfs_weights = defaultdict(int)
        for lm_id, _ in self.last_lk:
            # find the number of sharing landmarks between
            # the current frame and each of the neighbor keyframes
            connected_kfs = self.graph[lm_id]
            for kfs in connected_kfs:
                kfs_weights[kfs] += 1

        print("kfs_weights: ", kfs_weights, len(kfs_weights))
        if len(kfs_weights) == 0:
            return

        self.local_keyframes.clear()
        max_weight = 0
        for kf_id, weight in kfs_weights.items():
            self.local_keyframes.append(kf_id)
            kf: Keyframe = self.graph.node[kf_id]['data']
            kf.local_map_update_identifier = frame.frame_id
            if weight > max_weight:
                max_weight = weight

    def update_local_keyframes(self, frame: Frame):
        """Count number of lm shared between current frame and neighbour KFs
        (count obs.). For each keyframe, we keep count of how many lms it
        shares with the current one.
        """
        print("update_local_keyframes")
        kfs_weights = defaultdict(int)
        for lm_id in frame.landmarks_:
            # find the number of sharing landmarks between
            # the current frame and each of the neighbor keyframes
            connected_kfs = self.graph[lm_id]
            for kfs in connected_kfs:
                kfs_weights[kfs] += 1

        print("kfs_weights: ", kfs_weights, len(kfs_weights))
        if len(kfs_weights) == 0:
            return

        self.local_keyframes.clear()
        max_weight = 0
        for kf_id, weight in kfs_weights.items():
            self.local_keyframes.append(kf_id)
            kf: Keyframe = self.graph.node[kf_id]['data']
            kf.local_map_update_identifier = frame.frame_id
            if weight > max_weight:
                max_weight = weight
                self.nearest_covisibility = kf

    def update_local_landmarks(self, frame: Frame):
        """Update local landmarks by adding
        all the landmarks of the local keyframes.
        """
        self.local_landmarks.clear()
        print("update_local_landmarks")
        for kf_id in self.local_keyframes:
            print("kf_id: ", kf_id)
            for lm_id in self.graph[kf_id]:
                if len(self.graph.nodes[str(lm_id)]) == 0:
                    print("Problem: ", lm_id)
                else:
                    lm = self.graph.node[str(lm_id)]['data']
                    # Avoid duplication
                    if lm.local_map_update_identifier == frame.frame_id:
                        continue
                    lm.local_map_update_identifier = frame.frame_id
                    self.local_landmarks.append(lm_id)
        print("self.local_landmarks: ",
              len(self.local_landmarks), len(self.local_keyframes))
        # count the number of lmid
        #TODO: This is for debug
        lm_count = defaultdict(int)
        for lm in self.local_landmarks:
            lm_count[lm] += 1
            if lm_count[lm] > 1:
                print("Double landmark: ", lm)
                exit()

    def apply_landmark_replace(self):
        print('apply landmark?')
        pass

    def set_local_landmarks(self):
        print("set_local_landmarks()")

    def update_local_map(self, frame: Frame):
        """Called after init and normal tracking
        """
        print("update_local_map for current frame: ",
              frame.frame_id, frame.im_name)
        # Todo: unify update_local_kf, keyframes and set
        self.update_local_keyframes(frame)
        self.update_local_landmarks(frame)
        return True
    
    def update_local_map_lk(self, frame: Frame):
        """Called after init and normal tracking
        """
        print("update_local_map for current frame: ",
              frame.frame_id, frame.im_name)
        # Todo: unify update_local_kf, keyframes and set
        self.update_local_keyframes_lk(frame)
        self.update_local_landmarks(frame)
        return True

    def search_local_landmarks_in_kf(self, kf: Keyframe):
        """Acquire more 2D-3D matches by reprojecting the 
        local landmarks to the current frame

        Return:
            - matches: Nx2 matrix with the [feature_id, landmark_id]
        """
        margin = 5
        print("self.local_landmarks: ", len(self.local_landmarks))
        if len(self.local_landmarks) == 0:
            return []
        matches = slam_matcher.\
            match_frame_to_landmarks(kf.descriptors, self.local_landmarks, margin,
                                     self.data, self.graph)
        print("matches: ", len(matches))
        return matches

    def fix_keyframes(self, n_fix_frames=1, max_kf_opt=5):
        """Compute KFs to be fixed/constant during optimizations
        
        Returns a boolean dictionary with im_name as key and
        True means fixed/constant (typically the first KFs)
        False should be optimized
        """
        kf_dict = {}
        n_keyframes = len(self.keyframes)
        print("self.n_keyframes", self.n_keyframes)
        print("self.keyframes", self.keyframes)
        if n_keyframes == 0 or max_kf_opt < 0:
            return []
        n_elem_const = max(n_keyframes-max_kf_opt, n_fix_frames)
        for kf_c in self.keyframes[0:n_elem_const]:
            kf_dict[kf_c] = True
        for kf_v in self.keyframes[n_elem_const:]:
            kf_dict[kf_v] = False
        return kf_dict

    def fix_local_keyframes(self, n_fix_frames=1, max_kf_opt=5):
        kf_dict = {}
        n_keyframes = len(self.local_keyframes)
        if n_keyframes == 0 or max_kf_opt < 0:
            return []
        n_elem_const = max(n_keyframes-max_kf_opt, n_fix_frames)
        for kf_c in self.local_keyframes[0:n_elem_const]:
            kf_dict[kf_c] = True
        for kf_v in self.local_keyframes[n_elem_const:]:
            kf_dict[kf_v] = False
        return kf_dict

    def update_lk_landmarks(self):
        """Update the lk landmarks after cleaning up the graph
        """
        print("lk_lms before: ", len(self.last_lk), self.curr_kf.im_name)
        self.last_lk.clear()
        for lm_id in self.graph[self.curr_kf.im_name]:
            edge = self.graph.get_edge_data(str(lm_id), self.curr_kf.im_name)
            self.last_lk.append((lm_id, np.hstack([edge['feature'], edge['feature_scale']])))

        print("lk_lms after: ", len(self.last_lk))
        # Check for double
        lms = defaultdict(int)
        for l in self.last_lk:
            print(l)
            lms[l[0]] += 1
            if lms[l[0]] > 1:
                print("double", l)
                exit()

    def local_bundle_adjustment2(self):
        """This is very similar to bundle_tracking
        The main difference is that we add a number
        of frames but "fix" the positions of the oldest.
        """
        print("local_bundle_adjustment! new version")
        # We can easily build the equation system from the reconstruction
        if self.n_keyframes <= 2:
            return
        ba = csfm.BundleAdjuster()
        c_lms = defaultdict(int)
        for lm_id in self.local_landmarks:
            c_lms[lm_id] += 1
            if c_lms[lm_id] > 1:
                print(" local_bundle_adjustment2 Double: ", lm_id) 
        for camera in self.reconstruction.cameras.values():
            reconstruction._add_camera_to_bundle(ba, camera, constant=True)
        
        # Find "earliest" KF seen by the current map! 
        n_kfs_fixed = 2
        n_kfs_optimize = 10
        # First, create a list of all frames and fix all but the newest N
        kf_constant = self.fix_keyframes(n_kfs_fixed, n_kfs_optimize)
        # TODO: There might be a problem if the landmarks have no connections to the fixed frames
        kf_added = {}
        n_observations = 0
        n_obs_rel = 0
        all_opt = True
        for lm_id in self.local_landmarks:
            lm_node = self.graph[lm_id]

            if (len(lm_node) > 2):
                n_obs_rel += len(lm_node) 
            # add lm
            point = self.reconstruction.points[lm_id]
            ba.add_point(point.id, point.coordinates, False)    
            for kf_id in lm_node.keys():
                k = kf_added.get(str(kf_id))
                # add kf if not added
                if k is None:
                    kf_added[str(kf_id)] = True
                    shot = self.reconstruction.shots[kf_id]
                    r = shot.pose.rotation
                    t = shot.pose.translation
                    ba.add_shot(shot.id, shot.camera.id,
                                r, t, kf_constant[str(kf_id)])
                    if kf_constant[str(kf_id)]:
                        all_opt = False
                # add observation
                scale = self.graph[kf_id][lm_id]['feature_scale']
                pt = self.graph[kf_id][lm_id]['feature']
                ba.add_point_projection_observation(
                    kf_id, lm_id, pt[0], pt[1], scale)
                n_observations += 1
        print("n_obs: ", n_observations)
        print("n_obs_rel: ", n_obs_rel)
        
        # try to find if the added frames are all optimizeable
        if all_opt:
            print("All frames optimized, please fix!")
            exit()
        config = self.data.config
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
        chrono = Chronometer()
        chrono.lap('setup')
        ba.run()
        chrono.lap('run_local_bundle_adjustment')
        # update frame poses!
        for kf_id in kf_added.keys():
            # don't update fixed ones
            if not kf_constant[kf_id]:
                # update reconstruction
                shot = self.reconstruction.shots[kf_id]
                s = ba.get_shot(shot.id)
                shot.pose.rotation = [s.r[0], s.r[1], s.r[2]]
                shot.pose.translation = [s.t[0], s.t[1], s.t[2]]
                # update kf                
                kf: Keyframe = self.graph.node[kf_id]['data']
                print("kf name: ", kf.im_name,
                      " prev: ", kf.world_pose.rotation, kf.world_pose.translation,
                      " curr: ", shot.pose.rotation, shot.pose.translation)
                kf.world_pose.rotation = [s.r[0], s.r[1], s.r[2]]
                kf.world_pose.translation = [s.t[0], s.t[1], s.t[2]]
        n_rem_nodes = 0
        # check duplicates
        c_lms = defaultdict(int)
        for lm_id in self.local_landmarks:
            c_lms[lm_id] += 1
            if c_lms[lm_id] > 1:
                print(" bef clean Double: ", lm_id) 

        for lm_id in self.local_landmarks:
            point = self.reconstruction.points[lm_id]
            p = ba.get_point(point.id)
            point.coordinates = [p.p[0], p.p[1], p.p[2]]
            point.reprojection_errors = p.reprojection_errors
            # print("p: ", p.reprojection_errors)
            self.clean_up_graph(lm_id, p.reprojection_errors)
            if not self.graph.has_node(lm_id):
                n_rem_nodes += 1
        chrono.lap('update')
        print("Removed {} from graph".format(n_rem_nodes))
        print("Local BA Times: ", chrono.lap_times())
        slam_debug.avg_timings.addTimes(chrono.laps_dict)
        logger.debug(ba.brief_report())
        report = {
            'wall_times': dict(chrono.lap_times()),
            'brief_report': ba.brief_report(),
        }
        print("Report: ", report)
        return report

    def local_bundle_adjustment(self):
        """This is very similar to bundle_tracking
        The main difference is that we add a number
        of frames but "fix" the positions of the oldest.
        """
        print("local_bundle_adjustment!")
        # We can easily build the equation system from the reconstruction
        if self.n_keyframes <= 2:
            return
        ba = csfm.BundleAdjuster()
        
        for camera in self.reconstruction.cameras.values():
            reconstruction._add_camera_to_bundle(ba, camera, constant=True)
        
        # Find "earliest" KF seen by the current map! 
        n_kfs_fixed = 2
        n_kfs_optimize = 10
        # First, create a list of all frames and fix all but the newest N
        kf_constant = self.fix_keyframes(n_kfs_fixed, n_kfs_optimize)
        # TODO: There might be a problem if the landmarks have no connections to the fixed frames
        kf_added = {}
        n_observations = 0
        n_obs_rel = 0
        all_opt = True
        for lm_id in self.local_landmarks:
            lm_node = self.graph[lm_id]

            if (len(lm_node) > 2):
                n_obs_rel += len(lm_node) 
            # add lm
            point = self.reconstruction.points[lm_id]
            ba.add_point(point.id, point.coordinates, False)    
            for kf_id in lm_node.keys():
                k = kf_added.get(str(kf_id))
                # add kf if not added
                if k is None:
                    kf_added[str(kf_id)] = True
                    shot = self.reconstruction.shots[kf_id]
                    r = shot.pose.rotation
                    t = shot.pose.translation
                    ba.add_shot(shot.id, shot.camera.id,
                                r, t, kf_constant[str(kf_id)])
                    if kf_constant[str(kf_id)]:
                        all_opt = False
                # add observation
                scale = self.graph[kf_id][lm_id]['feature_scale']
                pt = self.graph[kf_id][lm_id]['feature']
                ba.add_point_projection_observation(
                    kf_id, lm_id, pt[0], pt[1], scale)
                n_observations += 1
        #try to find if the added frames are all optimizeable
        if all_opt:
            print("All frames optimized, please fix!")
            exit()
        config = self.data.config
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
        chrono = Chronometer()
        chrono.lap('setup')
        ba.run()
        chrono.lap('run_local_bundle_adjustment')
        # update frame poses!
        for kf_id in kf_added.keys():
            # don't update fixed ones
            if not kf_constant[kf_id]:
                # update reconstruction
                shot = self.reconstruction.shots[kf_id]
                s = ba.get_shot(shot.id)
                shot.pose.rotation = [s.r[0], s.r[1], s.r[2]]
                shot.pose.translation = [s.t[0], s.t[1], s.t[2]]
                # update kf                
                kf: Keyframe = self.graph.node[kf_id]['data']
                kf.world_pose.rotation = [s.r[0], s.r[1], s.r[2]]
                kf.world_pose.translation = [s.t[0], s.t[1], s.t[2]]
        n_rem_nodes = 0
        for lm_id in self.local_landmarks:
            point = self.reconstruction.points[lm_id]
            p = ba.get_point(point.id)
            point.coordinates = [p.p[0], p.p[1], p.p[2]]
            point.reprojection_errors = p.reprojection_errors
            self.clean_up_graph(lm_id, p.reprojection_errors)
            if not self.graph.has_node(lm_id):
                n_rem_nodes += 1
        chrono.lap('update')
        print("Removed {} from graph".format(n_rem_nodes))
        print("Local BA Times: ", chrono.lap_times())
        slam_debug.avg_timings.addTimes(chrono.laps_dict)
        logger.debug(ba.brief_report())
        report = {
            'wall_times': dict(chrono.lap_times()),
            'brief_report': ba.brief_report(),
        }
        print("Report: ", report)
        return report

    def clean_up_graph(self, lm_id, reprojection_errors: dict):
        """Removes outlier observations from the graph and if one or no observations are left
        remove also the landmark from the graph
        """
        n_th = 0
        th = 0.006
        for (k, v) in reprojection_errors.items():
            if np.linalg.norm(v) > th:
                print("remove lm_id: ", type(lm_id), lm_id, " k: ", k)
                # remove outlier observations
                self.graph.remove_edge(lm_id, k)
                # k -> kf_id
                n_th += 1
        # Remove lm if just one or no observations
        if len(self.graph[lm_id]) <= 1:
            self.graph.remove_node(lm_id)
            print("remove lm: ", lm_id)
            del self.reconstruction.points[lm_id]
        else:
            # Remove observation
            lm = self.graph.node[lm_id]['data']
            lm.num_observed -= n_th
            print("lm_num_observed: ", lm_id, lm.num_observed)

    def search_local_landmarks(self, frame: Frame):
        """ Acquire more 2D-3D matches by reprojecting the 
        local landmarks to the current frame.
        """
        margin = 5
        print("frame.landmarks_: ", len(frame.landmarks_))
        print("self.local_landmarks: ", len(self.local_landmarks))
        frame.landmarks_[:] = list(set(frame.landmarks_+self.local_landmarks))
        print("after frame.landmarks_: ", len(frame.landmarks_))
        matches = slam_matcher.\
            match_frame_to_landmarks(frame.descriptors, frame.landmarks_, margin,
                                     self.data, self.graph)
        print("len(matches): ", len(matches))
        # Let's assume that matches are mostly correct and matched landmarks are visible!
        if matches is None:
            return None
        for _, flm_id in matches:
            lm_id = frame.landmarks_[flm_id]
            lm = self.graph.node[lm_id]['data']
            lm.num_observable += 1
        print("matches: ", len(matches))
        return np.asarray(matches)

    def observable_in_frame(self, frame: Frame):
        """ Similar to frame.can_observe in OpenVSlam
        """
        pose_world_to_cam = frame.world_pose
        cam_center = frame.world_pose.get_origin()
        factor = self.camera[1].height/self.camera[1].width
        observations = []
        for lm_id in self.local_landmarks:
            lm = self.graph.node[lm_id]['data']
            if lm.identifier_in_local_lm_search_ == frame.frame_id:
                continue
            # check if observeable
            p = self.reconstruction.points[str(lm.lm_id)].coordinates
            camera_point = pose_world_to_cam.transform(p)
            print("camera_point", camera_point)
            if camera_point[2] <= 0.0:
                continue
            point2D = self.camera[1].project(camera_point)
            is_in_image = slam_utils.in_image(point2D, factor)
            #TODO: check boundaries?
            cam_to_lm_vec = p - cam_center
            cam_to_lm_dist = np.linalg.norm(cam_to_lm_vec)
            #TODO: Check feature scale?
            # Compute normal
            lm.update_normal_and_depth(p, self.graph)
            mean_normal = lm.mean_normal
            ray_cos = np.dot(cam_to_lm_vec, mean_normal)/cam_to_lm_dist
            if ray_cos < 0.5:
                continue
            observations.append(point2D)
        return observations

    # OpenVSlam mapping module
    def mapping_with_new_keyframe(self, curr_kfm: Keyframe):
        """
        - Removes redundant frames
        - Creates new!! landmarks create_new_landmarks()
        - updates keyframe
        """
        if self.curr_kf is not None:
            old_frame = self.curr_kf.im_name
        else:
            old_frame = ""
        self.curr_kf = curr_kfm
        print("mapping_with_new_keyframe", curr_kfm.im_name,
              ", ", old_frame, self.curr_kf.im_name)

        # Store the landmarks seen in frame 2
        self.store_new_keyframe()
        in_graph = {}
        frame1 = self.curr_kf.im_name
        seen_landmarks = self.graph[frame1]
        print("frame1: ", frame1)
        for lm_id in seen_landmarks:
            e = self.graph.get_edge_data(frame1, lm_id)
            if e['feature_id'] in in_graph:
                print("Already in there mapping after store!", e['feature_id'],
                      "lm_id: ", lm_id)
                exit()
            in_graph[e['feature_id']] = lm_id
        print("loc lms: ", len(self.local_landmarks))
        # remove redundant landmarks
        self.remove_redundant_landmarks()
        print("loc lms1: ", len(self.local_landmarks))
        self.create_new_observations_for_lm(self.data)
        print("loc lms2: ", len(self.local_landmarks))
        self.create_new_landmarks(self.data)
        print("loc lms3: ", len(self.local_landmarks))
        self.fuse_duplicated_landmarks()
        print("loc lms4: ", len(self.local_landmarks))

    # OpenVSlam mapping module
    def mapping_with_new_keyframe_lk(self, curr_kfm: Keyframe):
        """
        - Removes redundant frames
        - Creates new!! landmarks create_new_landmarks()
        - updates keyframe
        """
        if self.curr_kf is not None:
            old_frame = self.curr_kf.im_name
        else:
            old_frame = ""
        self.curr_kf = curr_kfm
        print("mapping_with_new_keyframe_lk", curr_kfm.im_name,
              ", ", old_frame, self.curr_kf.im_name)

        self.remove_redundant_landmarks()
        print("loc lms1: ", len(self.local_landmarks))
        self.create_new_observations_for_lm(self.data)
        print("loc lms2: ", len(self.local_landmarks))
        self.create_new_landmarks(self.data)
        print("loc lms3: ", len(self.local_landmarks))
        self.fuse_duplicated_landmarks()
        print("loc lms4: ", len(self.local_landmarks))

    def create_new_observations_for_lm(self, data):
        """Creates an observation (edge) between landmarks and kfs
        for old landmarks.
        """
        print("create_new_observations_for_lm: {} with len(local_landmarks): {}".
              format(self.curr_kf.im_name, len(self.local_landmarks)))
        # first match all the local landmarks to the featues in self.curr_kf
        matches_lm_f = self.search_local_landmarks_in_kf(self.curr_kf)

        chrono = Chronometer()
        print("len(matches_lm_f): ", len(matches_lm_f))
        if self.feature_ids_last_frame is not None:
            print("len: ", len(self.feature_ids_last_frame))
        p1 = self.curr_kf.points
        c1 = self.curr_kf.colors
        n_added = 0
        for (f1_id, loc_lm_id) in matches_lm_f:
        # for (f1_id, loc_lm_id) in self.feature_ids_last_frame.items():
            if self.curr_kf.matched_lms[f1_id] != -1:
                if self.curr_kf.im_name == "000002.png":
                    print("Already matched in current frame", self.graph[str(self.curr_kf.matched_lms[f1_id])])
                    lm = self.graph.node[str(self.curr_kf.matched_lms[f1_id])]['data']
                    lm_2 = self.local_landmarks[loc_lm_id]
                    print("lm: ", lm.lm_id, " lm2: ", lm_2, " n_obs: ", lm.num_observed, lm.num_observable, lm.get_observed_ratio())
                continue
            lm_id = self.local_landmarks[loc_lm_id]

            lm: Landmark = self.graph.node[lm_id]['data']
            x, y, s = p1[f1_id, 0:3]
            r, g, b = c1[f1_id, :]

            if self.graph.has_node(lm_id):
                # add observations
                self.graph.add_edge(self.curr_kf.im_name, lm_id,
                                    feature=(float(x), float(y)),
                                    feature_scale=float(s),
                                    feature_id=int(f1_id),
                                    feature_color=(float(r), float(g), float(b)))
                pos_w = self.reconstruction.points[lm_id].coordinates
                lm.update_normal_and_depth(pos_w, self.graph)
                lm.compute_descriptor(self.curr_kf, self.graph)
                self.curr_kf.matched_lms[f1_id] = lm_id
                n_added += 1
                lm.num_observed += 1
                if not self.config_slam['tracker_lk']:
                    lm.num_observable += 1
                print("lm: ", lm_id, " obs: ", lm.get_observed_ratio(), lm.num_observed)
        print("added {} new observations to graph for {} ".
              format(n_added, self.curr_kf.im_name))

    def create_new_landmarks(self, data):
        num_covisibilites = 10
        #TODO: get top n covisibilites
        curr_cam_center = self.curr_kf.world_pose.get_origin()
        #If match with landmark, add an observation
        cov_frames = self.covisibility_list[-num_covisibilites:]
        for neighbor_kfm in cov_frames:
            if neighbor_kfm == self.curr_kf.im_name:
                continue
            print("Trying to triangulate: ",
                  neighbor_kfm, "<->", self.curr_kf.im_name)
            n_kfm = self.graph.nodes[neighbor_kfm]['data']
            print("create_new_landmarks neighbor_kfm: ", neighbor_kfm, n_kfm)
            kf_cam_center = n_kfm.world_pose.get_origin()
            baseline = kf_cam_center - curr_cam_center
            dist = np.linalg.norm(baseline)
            median_depth = n_kfm.\
                compute_median_depth(True, self.graph, self.reconstruction)
            if dist < 0.02 * median_depth:
                continue

            #match the top 10 frames!
            chrono = Chronometer()
            matches = slam_matcher.\
                match_for_triangulation(self.curr_kf, n_kfm,
                                        self.graph, self.data)
            chrono.lap("match_tri")
            slam_debug.avg_timings.addTimes(chrono.laps_dict)
            self.triangulate_with_two_kfs(self.curr_kf, n_kfm, matches, data)
        
        self.fuse_duplicated_landmarks()
        if self.config_slam['tracker_lk']:
            self.acquire_new_observations(self.curr_kf)
        return True

    def acquire_new_observations(self, curr_kf: Keyframe):
        # return
        # Now, we triangulated a bunch of landmarks and now have to match them with other frames
        # It's easy to imagine like that:
        # - Current frame is F2
        # - Other frames are F1 and F0
        # Matching F2<->F0 and triangulating also defines the ids
        # We will add most of the observations for F1 by matching
        # but we also create new features. Thus, we have to match those to all the others
        lms = self.graph[curr_kf.im_name]
        print("acquire_new_observations: ", len(lms))
        for l in lms:
            print("l: ", l, " kf: ", curr_kf.im_name, ": ", self.graph[l]) #self.graph.get_edge_data(l,curr_kf.im_name))
        # Get all the LMS from the graph
        for other_kf in self.local_keyframes:
            if other_kf == curr_kf.im_name:
                continue
            kf = self.graph.node[other_kf]['data']
            p1, f1, c1 = kf.load_points_desc_colors()
            match_lms = []
            f2 = []
            for lm_id in lms:
                if len(self.graph[lm_id]) < len(self.local_keyframes):
                    print("Checking {} for landmark {} with {}".format(other_kf, lm_id,len(self.graph[lm_id])))
                    print(self.graph[lm_id])
                if not self.graph.has_edge(other_kf, lm_id): # Already matched
                    match_lms.append(lm_id)
                    f2.append(self.graph.node[lm_id]['data'].descriptor)
            print("Trying to match {} lms to {} features".format(len(f2),len(f1)))
            # Now match
            matches = slam_matcher.match_desc_desc(f1, np.asarray(f2), self.data) #, self.graph)
            print("Found {} new observations between {} <-> {}".format(len(matches), curr_kf.im_name, other_kf))
            for idx1, idx2 in matches:
                lm_id = match_lms[idx2]
                lm = self.graph.node[lm_id]['data']
                x, y, s = p1[idx1, 0:3]
                r, g, b = c1[idx1, :]
                print("has edge: ", other_kf, lm_id, self.graph.has_edge(other_kf, lm_id), self.graph.has_edge(lm_id, other_kf))
                # if kf.matched_lms[idx1] != -1:
                #     print("Already matched to ", kf.matched_lms[idx1])
                #     # match in other image
                #     e = self.graph.get_edge_data(str(kf.matched_lms[idx1]), curr_kf.im_name) #['feature']
                #     e2 = self.graph.get_edge_data(lm_id, curr_kf.im_name)
                #     print("e: ", e)
                #     print("e2: ", e2)
                #     p_m = np.array([np.hstack([e['feature'],e['feature_scale'],0]), np.hstack([e2['feature'],e2['feature_scale'],0])])
                #     # e['feature'],e['feature_scale']
                #     # visualize
                #     slam_debug.disable_debug = False
                #     im1 = self.data.load_image(curr_kf.im_name)
                #     im2 = self.data.load_image(other_kf)
                #     slam_debug.visualize_matches_pts(p_m, np.vstack([p1[idx1, :], p1[idx1, :]]), np.array([[0,0],[1,1]]),im1,im2,True)
                #     e = self.graph.get_edge_data(str(kf.matched_lms[idx1]), '00005.png') #['feature']
                #     e2 = self.graph.get_edge_data(lm_id, '00005.png')
                #     p_m = np.array([np.hstack([e['feature'],e['feature_scale'],0]), np.hstack([e2['feature'],e2['feature_scale'],0])])
                #     slam_debug.visualize_matches_pts(p_m, np.vstack([p1[idx1, :], p1[idx1, :]]), np.array([[0,0],[1,1]]),im1,im2,True)
                #     slam_debug.disable_debug = True
                # add observations
                self.graph.add_edge(other_kf, lm_id,
                                    feature=(float(x), float(y)),
                                    feature_scale=float(s),
                                    feature_id=int(idx1),
                                    feature_color=(float(r), float(g), float(b)))
                print("create observation: ",  self.graph.get_edge_data(other_kf, lm_id))
                pos_w = self.reconstruction.points[lm_id].coordinates
                lm.update_normal_and_depth(pos_w, self.graph)
                lm.compute_descriptor(kf, self.graph)
                kf.matched_lms[idx1] = lm_id
                lm.num_observed += 1
            self.fuse_duplicated_landmarks()


    def triangulate_with_two_kfs(self, kf1: Keyframe, kf2: Keyframe, matches, data):
        """kf1 -> current frame
        kf2 -> frame to triangulate with 
        """
        if matches is None:
            return
        chrono = Chronometer()
        frame1, frame2 = kf1.im_name, kf2.im_name
        p1, f1, c1 = kf1.load_points_desc_colors()
        p2, f2, c2 = kf2.load_points_desc_colors()
        chrono.lap("loading_tri")
        # Now, build up the graph for the triangulation
        chrono.start()

        # TODO: try without tracks graph
        # create the graph
        tracks_graph = nx.Graph()
        tracks_graph.add_node(str(frame1), bipartite=0)
        tracks_graph.add_node(str(frame2), bipartite=0)
        f_processed = defaultdict(int)
        for (track_id, (f1_id, f2_id)) in enumerate(matches):
            # this checks whether the current kf was matched
            # to one of the landmarks.
            # if f2 is already in a lm
            f_processed[f1_id] += 1
            if f_processed[f1_id] > 1:
                print("double add!!")
                exit()
            if kf1.matched_lms[f1_id] == -1:  # Check if match already belongs to a landmark
                # if not matched yet, assign the landmark of kf2
                old_lm_id = kf2.matched_lms[f2_id]  
                if old_lm_id != -1:
                    print("old_lm_id: ", old_lm_id, self.graph.has_node(old_lm_id), self.graph.has_node(str(old_lm_id)))
                    if self.graph.has_node(str(old_lm_id)):
                        # also add the new track
                        print("Not matched in current frame but matched to other frame!", old_lm_id,
                                kf1.matched_lms[f1_id])
                        print("new! triang: track_id  {}, frames: {}<->{} f1_id {}, f2_id {}".
                            format(old_lm_id, frame1, frame2, f1_id, f2_id))
                        x, y, s = p1[f1_id, 0:3]
                        r, g, b = c1[f1_id, :]
                        self.graph.add_edge(str(frame1),
                                            str(old_lm_id),
                                            feature=(float(x), float(y)),
                                            feature_scale=float(s),
                                            feature_id=int(f1_id),
                                            feature_color=(float(r), float(g), float(b)))
                        kf1.matched_lms[f1_id] = old_lm_id
                        lm = self.graph.node[str(old_lm_id)]['data']
                        lm.num_observed += 1
                        lm.num_observable += 1
                else: # found a match not triangulated yet
                    x, y, s = p2[f2_id, 0:3]
                    r, g, b = c2[f2_id, :]
                    tracks_graph.add_node(str(track_id), bipartite=1)
                    tracks_graph.add_edge(str(frame2),
                                          str(track_id),
                                          feature=(float(x), float(y)),
                                          feature_scale=float(s),
                                          feature_id=int(f2_id),
                                          feature_color=(float(r), float(g), float(b)))

                    x, y, s = p1[f1_id, 0:3]
                    r, g, b = c1[f1_id, :]
                    tracks_graph.add_edge(str(frame1),
                                          str(track_id),
                                          feature=(float(x), float(y)),
                                          feature_scale=float(s),
                                          feature_id=int(f1_id),
                                          feature_color=(float(r), float(g), float(b)))
        chrono.lap("track_graph")
        cameras = data.load_camera_models()
        camera = next(iter(cameras.values()))
        rec_tri = types.Reconstruction()
        rec_tri.reference = data.load_reference()
        rec_tri.cameras = cameras

        shot1 = types.Shot()
        shot1.id = frame1
        shot1.camera = camera
        shot1.pose = kf1.world_pose
        shot1.metadata = reconstruction.get_image_metadata(data, frame1)
        rec_tri.add_shot(shot1)

        shot2 = types.Shot()
        shot2.id = frame2
        shot2.camera = camera
        shot2.pose = kf2.world_pose
        shot2.metadata = reconstruction.get_image_metadata(data, frame2)
        rec_tri.add_shot(shot2)

        graph_inliers = nx.Graph()
        chrono.lap("ba setup")
        np_before = len(rec_tri.points)
        reconstruction.triangulate_shot_features(tracks_graph, graph_inliers,
                                                 rec_tri, frame1,
                                                 data.config)
        np_after = len(rec_tri.points)
        chrono.lap("triangulateion")
        print("Triangulation times: ", chrono.lap_times())
        slam_debug.avg_timings.addTimes(chrono.laps_dict)
        print("Points before triangulation {} and after {} ".format(np_before, np_after))
        edges1 = graph_inliers.edges(frame1)
        points = rec_tri.points
        points3D = np.zeros((len(points), 3))
        for idx, pt3D in enumerate(points.values()):
            points3D[idx, :] = pt3D.coordinates
        # Due to some sorting issues, we have to go through
        matches_dbg = np.zeros([len(graph_inliers.edges(frame1)), 2], dtype=int)
        idx = 0
        # TODO: debug stuff, remove
        c_lms = defaultdict(int)
        for lm_id in self.local_landmarks:
            c_lms[lm_id] += 1
            if c_lms[lm_id] > 1:
                print(" before create Double: ", lm_id, frame1) 
        print("n lms: ", self.n_landmarks)
        n_bef_lms = self.n_landmarks
        # graph_inliers by "frames" first
        for _, gi_lm_id in graph_inliers.edges(frame1):
            # TODO: Write something like create_landmark
            lm_id = str(self.current_lm_id)
            lm = Landmark(lm_id)
            # print("Creating new lm: ", lm_id)
            lm.first_kf_id = kf1.kf_id
            self.n_landmarks += 1
            self.current_lm_id += 1
            # This is essentially the same as adding it to the graph
            self.add_landmark(lm)
            # Now, relate the gi_lm_id to the actual feature_id
            e1 = graph_inliers.get_edge_data(frame1, gi_lm_id)
            e2 = graph_inliers.get_edge_data(frame2, gi_lm_id)
            
            self.graph.add_edges_from([(frame1, str(lm_id), e1)])
            self.graph.add_edges_from([(frame2, str(lm_id), e2)])

            # also add the observations
            kf1.matched_lms[e1['feature_id']] = lm_id
            kf2.matched_lms[e2['feature_id']] = lm_id
            matches_dbg[idx, :] = np.array([e1['feature_id'], e2['feature_id']])
            idx += 1
            lm.compute_descriptor(kf1, self.graph)
            lm.update_normal_and_depth(pt3D.coordinates, self.graph)
            # We also have to add the points to the reconstruction
            point = types.Point()
            point.id = str(lm_id)
            point.coordinates = rec_tri.points[gi_lm_id].coordinates
            self.reconstruction.add_point(point)
            self.local_landmarks.append(lm.lm_id)
            self.add_fresh_landmark(lm.lm_id)
            # only +1 because already init with 1
            lm.num_observed += 1
            lm.num_observable += 1

        print("created {} for a total of lms {}".format(self.n_landmarks-n_bef_lms, self.n_landmarks))
        # TODO: debug stuff, remove
        c_lms = defaultdict(int)
        for lm_id in self.local_landmarks:
            c_lms[lm_id] += 1
            if c_lms[lm_id] > 1:
                print(" after create Double: ", lm_id, frame1) 
        # TODO: debug stuff, remove
        if (len(matches_dbg) > 0 and False):
            print("Newly created landmarks!")
            # def visualize_matches_pts(pts1, pts2, matches, im1, im2, do_show=True, title = ""):
            im1, im2 = data.load_image(kf1.im_name), data.load_image(kf2.im_name)
            slam_debug.visualize_matches_pts(p1, p2, matches_dbg, im1, im2, False, "matches_tri" )#, frame1, frame2, data, False)
            points3D_debug = np.zeros([len(rec_tri.points), 3])
            for idx, p in enumerate(rec_tri.points.values()):
                points3D_debug[idx, :] = p.coordinates
            slam_debug.reproject_landmarks(points3D_debug, np.zeros([len(rec_tri.points), 2]),
                                        kf2.world_pose, kf2.im_name, camera,
                                        self.data, title="triangulated: "+kf2.im_name, do_show=True)  
        print("test")

    # def triangulate_with_two_kfs(self, kf1: Keyframe, kf2: Keyframe, matches, data):
    #     """kf1 -> current frame
    #     kf2 -> frame to triangulate with 
    #     """
    #     if matches is None:
    #         return
    #     chrono = Chronometer()
    #     frame1, frame2 = kf1.im_name, kf2.im_name
    #     # p1, f1, c1 = feature_loader.instance.load_points_features_colors(
    #                 #  data, frame1, masked=True)
    #     # p2, f2, c2 = feature_loader.instance.load_points_features_colors(
    #                 #  data, frame2, masked=True)
    #     p1, f1, c1 = kf1.load_points_desc_colors() # kf1.points, kf1.descriptors, kf1.colors
    #     p2, f2, c2 = kf2.load_points_desc_colors() # kf2.points, kf2.descriptors, kf2colors
        
    #     chrono.lap("loading_tri")
    #     # Now, build up the graph for the triangulation
    #     chrono.start()
    #     # create the graph
    #     tracks_graph = nx.Graph()
    #     tracks_graph.add_node(str(frame1), bipartite=0)
    #     tracks_graph.add_node(str(frame2), bipartite=0)

    #     for (track_id, (f1_id, f2_id)) in enumerate(matches):
    #         # this checks whether the current kf was matched
    #         # to one of the landmarks.
    #         # if f2 is already in a lm
    #         if kf1.matched_lms[f1_id] == -1:
    #             old_lm_id = kf2.matched_lms[f2_id]
    #             if old_lm_id != -1:
    #                 if self.graph.has_node(old_lm_id):
    #                     # also add the new track
    #                     print("Not matched in current frame but matched to other frame!", old_lm_id,
    #                             kf1.matched_lms[f1_id])
    #                     print("new! triang: track_id  {}, frames: {}<->{} f1_id {}, f2_id {}".
    #                         format(old_lm_id, frame1, frame2, f1_id, f2_id))
    #                     x, y, s = p1[f1_id, 0:3]
    #                     r, g, b = c1[f1_id, :]
    #                     self.graph.add_edge(str(frame1),
    #                                         str(old_lm_id),
    #                                         feature=(float(x), float(y)),
    #                                         feature_scale=float(s),
    #                                         feature_id=int(f1_id),
    #                                         feature_color=(float(r), float(g), float(b)))
    #                     kf1.matched_lms[f1_id] = old_lm_id
    #             else:
    #                 x, y, s = p2[f2_id, 0:3]
    #                 r, g, b = c2[f2_id, :]
    #                 tracks_graph.add_node(str(track_id), bipartite=1)
    #                 tracks_graph.add_edge(str(frame2),
    #                                       str(track_id),
    #                                       feature=(float(x), float(y)),
    #                                       feature_scale=float(s),
    #                                       feature_id=int(f2_id),
    #                                       feature_color=(float(r), float(g), float(b)))

    #                 x, y, s = p1[f1_id, 0:3]
    #                 r, g, b = c1[f1_id, :]
    #                 tracks_graph.add_edge(str(frame1),
    #                                       str(track_id),
    #                                       feature=(float(x), float(y)),
    #                                       feature_scale=float(s),
    #                                       feature_id=int(f1_id),
    #                                       feature_color=(float(r), float(g), float(b)))
    #     chrono.lap("track_graph")
    #     cameras = data.load_camera_models()
    #     camera = next(iter(cameras.values()))
    #     rec_tri = types.Reconstruction()
    #     rec_tri.reference = data.load_reference()
    #     rec_tri.cameras = cameras

    #     shot1 = types.Shot()
    #     shot1.id = frame1
    #     shot1.camera = camera
    #     shot1.pose = kf1.world_pose
    #     shot1.metadata = reconstruction.get_image_metadata(data, frame1)
    #     rec_tri.add_shot(shot1)

    #     shot2 = types.Shot()
    #     shot2.id = frame2
    #     shot2.camera = camera
    #     shot2.pose = kf2.world_pose
    #     shot2.metadata = reconstruction.get_image_metadata(data, frame2)
    #     rec_tri.add_shot(shot2)

    #     graph_inliers = nx.Graph()
    #     chrono.lap("ba setup")
    #     np_before = len(rec_tri.points)
    #     reconstruction.triangulate_shot_features(tracks_graph, graph_inliers,
    #                                              rec_tri, frame1,
    #                                              data.config)
    #     np_after = len(rec_tri.points)
    #     chrono.lap("triangulateion")
    #     print("Triangulation times: ", chrono.lap_times())
    #     slam_debug.avg_timings.addTimes(chrono.laps_dict)
    #     print("Points before: {} and {} ".format(np_before, np_after))
    #     # visualize landmarks 2D points in KF <-> 2D points in new KF
    #     # and also reprojections!
    #     # draw triangulate features in im1
    #     # get observations
    #     edges1 = graph_inliers.edges(frame1)
    #     # we have the edges
    #     # try to find the same feature already existing in the graph!
    #     # n_duplicates = 0
    #     # for u, v in edges1:
    #     #     feature_id = graph_inliers.get_edge_data(u, v)['feature_id']
    #     #     for lm_id in self.graph[frame1]:
    #     #         feature_id2 = self.graph.\
    #     #             get_edge_data(frame1, lm_id)['feature_id']
    #     #         # if feature_id == feature_id2:
    #     #         #     print("created feature already in graph",
    #     #         #           feature_id, "<->", feature_id2)
    #     #         #     print("u,v", u, v)
    #     #         #     print("frame1", frame1, "lm_id", lm_id)
    #     #         #     print(self.graph[lm_id])
    #     #         #     n_duplicates += 1
    #     #         #     exit()
    #     # print("Created landmarks ", np_after, " with ",
    #     #       n_duplicates, " duplicates.")

    #     points = rec_tri.points
    #     points3D = np.zeros((len(points), 3))
    #     for idx, pt3D in enumerate(points.values()):
    #         points3D[idx, :] = pt3D.coordinates
    #     # Due to some sorting issues, we have to go through
    #     matches_dbg = np.zeros([len(graph_inliers.edges(frame1)), 2], dtype=int)
    #     idx = 0
    #     # graph_inliers by "frames" first
    #     for _, gi_lm_id in graph_inliers.edges(frame1):
    #         lm_id = str(self.current_lm_id)
    #         lm = Landmark(lm_id)
    #         lm.first_kf_id = kf1.kf_id
    #         self.n_landmarks += 1
    #         self.current_lm_id += 1
    #         # This is essentially the same as adding it to the graph
    #         self.add_landmark(lm)
    #         # Now, relate the gi_lm_id to the actual feature_id
    #         e1 = graph_inliers.get_edge_data(frame1, gi_lm_id)
    #         e2 = graph_inliers.get_edge_data(frame2, gi_lm_id)
            
    #         self.graph.add_edges_from([(frame1, str(lm_id), e1)])
    #         self.graph.add_edges_from([(frame2, str(lm_id), e2)])

    #         # also add the observations
    #         kf1.matched_lms[e1['feature_id']] = lm_id
    #         kf2.matched_lms[e2['feature_id']] = lm_id
    #         matches_dbg[idx, :] = np.array([e1['feature_id'], e2['feature_id']])
    #         idx += 1
    #         lm.compute_descriptor(self.graph)
    #         lm.update_normal_and_depth(pt3D.coordinates, self.graph)
    #         # We also have to add the points to the reconstruction
    #         point = types.Point()
    #         point.id = str(lm_id)
    #         point.coordinates = rec_tri.points[gi_lm_id].coordinates
    #         self.reconstruction.add_point(point)
    #         self.local_landmarks.append(lm.lm_id)
    #         self.add_fresh_landmark(lm.lm_id)

    #     if (len(matches_dbg) > 0):
    #         print("Newly created landmarks!")
    #         slam_debug.visualize_matches(matches_dbg, frame1, frame2, data, False)
    #         points3D_debug = np.zeros([len(rec_tri.points), 3])
    #         for idx, p in enumerate(rec_tri.points.values()):
    #             points3D_debug[idx, :] = p.coordinates
    #         slam_debug.reproject_landmarks(points3D_debug, np.zeros([len(rec_tri.points), 2]),
    #                                     kf2.world_pose, kf2.im_name, camera,
    #                                     self.data, title="triangulated: "+kf2.im_name, do_show=True)

    def remove_redundant_landmarks(self):
        observed_ratio_thr = 0.3
        num_reliable_keyfrms = 2
        num_obs_thr = 2
        lm_not_clear = 0
        lm_valid = 1
        lm_invalid = 2
        fresh_landmarks = self.fresh_landmarks
        num_removed = 0
        removed_landmarks = []
        cleaned_landmarks = []
        print("len(fresh_landmarks): ", len(fresh_landmarks))
        for lm_id in fresh_landmarks:
            lm_state = lm_not_clear
            if not self.graph.has_node(lm_id):
                removed_landmarks.append(lm_id)
                continue

            lm: Landmark = self.graph.node[lm_id]['data']
            num_observations = len(self.graph[lm_id])
            if lm.get_observed_ratio() < observed_ratio_thr:
                # if `lm` is not reliable
                # remove `lm` from the buffer and the database
                lm_state = lm_invalid
                print("lm {} invalid due to obs_ratio {}/{} = {} < {}".
                      format(lm_id, lm.num_observed, lm.num_observable, lm.get_observed_ratio(), observed_ratio_thr))
            elif num_reliable_keyfrms + lm.first_kf_id <= self.curr_kf.kf_id \
                    and num_observations <= num_obs_thr:
                # if the number of the observers of `lm` is small after some
                # keyframes were inserted
                # remove `lm` from the buffer and the database
                lm_state = lm_invalid
                print("lm {} invalid due rel. kfs {} + {} <= {} and {} <= {}".
                      format(lm_id, num_reliable_keyfrms, lm.first_kf_id,
                       self.curr_kf.kf_id, num_observations, num_obs_thr))

            elif num_reliable_keyfrms + 1 + lm.first_kf_id <= self.curr_kf.kf_id:
                # if the number of the observers of `lm` is small after some
                # keyframes were inserted
                # remove `lm` from the buffer and the database
                lm_state = lm_valid
            #     print("lm {} valid due rel. kfs {} + 1 + {} <= {}".
            #           format(lm_id, num_reliable_keyfrms, lm.first_kf_id, self.curr_kf.kf_id))
            # else:
            #     print("lm {} default unclear due rel. kfs {} + 1 + {} <= {}".
            #           format(lm_id, num_reliable_keyfrms, lm.first_kf_id, self.curr_kf.kf_id))

            if lm_state == lm_invalid:
                if self.feature_ids_last_frame is not None:
                    k = self.feature_ids_last_frame.get(str(lm_id))
                    if k is not None:
                        print("{} found in current frame!".format(lm_id))
                removed_landmarks.append(lm_id)
            else:
                cleaned_landmarks.append(lm_id)
 
        for lm_id in removed_landmarks:
            if self.graph.has_node(lm_id):
                self.graph.remove_node(lm_id)
                num_removed += 1
                del self.reconstruction.points[lm_id]
                
        print("remove_landmark: ", num_removed, len(removed_landmarks), " cleaned_landmarks: ", len(cleaned_landmarks))

        # somehow also remove from frame.landmarks_
        # clean-up frame.landmarks_
        keep_idx = np.zeros(len(self.curr_kf.landmarks_), dtype=bool)
        for idx, lm_id in enumerate(self.curr_kf.landmarks_):
            if self.graph.has_node(lm_id):
               keep_idx[idx] = True
            #    print("keep: ", idx, " lm_id: ", lm_id)
            
        self.curr_kf.landmarks_[:] = compress(self.curr_kf.landmarks_, keep_idx)

        keep_idx = np.zeros(len(self.local_landmarks), dtype=bool)
        for idx, lm_id in enumerate(self.local_landmarks):
            if self.graph.has_node(lm_id):
               keep_idx[idx] = True
        c_lms = defaultdict(int)
        for lm_id in self.local_landmarks:
            c_lms[lm_id] += 1
            if c_lms[lm_id] > 1:
                print(" bef red Double: ", lm_id) 

        self.local_landmarks[:] = compress(self.local_landmarks, keep_idx)
        c_lms = defaultdict(int)
        for lm_id in self.local_landmarks:
            c_lms[lm_id] += 1
            if c_lms[lm_id] > 1:
                print(" after red Double: ", lm_id) 
        keep_idx = np.zeros(len(self.last_frame.landmarks_), dtype=bool)
        for idx, lm_id in enumerate(self.last_frame.landmarks_):
            if self.graph.has_node(lm_id):
               keep_idx[idx] = True
        
        self.last_frame.landmarks_[:] = compress(self.last_frame.landmarks_, keep_idx)

        print("len: curr_kf: {}, local_lms: {}, landmarks: {}".
              format(len(self.curr_kf.landmarks_),
                     len(self.local_landmarks),
                     len(self.last_frame.landmarks_)))

    # def clean_landmarks(self, landmarks):
    #     """Removes landmarks that are not in the graph

    #     Returns a cleaned list of landmarks
    #     """
    #     cleaned_landmarks = []
    #     for lm_id in landmarks:
    #         if self.graph.has_node(lm_id):
    #             cleaned_landmarks.append(lm_id)
    #     return cleaned_landmarks

    def store_new_keyframe(self):
        curr_lms = self.curr_kf.landmarks_
        print("store_new_keyframe kf {} with {} landmarks: ".format(len(curr_lms), self.curr_kf.im_name))
        p, f, c = self.curr_kf.load_points_desc_colors()
        n_new_edges = 0
        for idx, lm_id in enumerate(curr_lms):
            lm: Landmark = self.graph.node[lm_id]['data']
            observations = self.graph[lm_id]
            if self.curr_kf.im_name in observations:
                self.add_fresh_landmark(lm_id)
            else:
                
                f1_id = self.feature_ids_last_frame[lm_id]
                # print("Already in graph? store", self.graph.get_edge_data(self.curr_kf.im_name, lm_id))
                if self.curr_kf.matched_lms[f1_id] != -1 or self.graph.get_edge_data(self.curr_kf.im_name, lm_id) is not None:
                    print("Adding an already matched edge!", f1_id, self.curr_kf.im_name, lm_id)
                    exit()
                # print(self.curr_kf.im_name, "Adding edge: ", " lm_id: ", lm_id, "f1_id: ", f1_id)
                self.curr_kf.matched_lms[f1_id] = lm_id
                x, y, s = p[f1_id, 0:3]
                r, g, b = c[f1_id, :]

                #TODO: add feature id
                if self.graph.has_node(lm_id):
                    self.graph.add_edge(self.curr_kf.im_name, lm_id,
                                        feature=(float(x), float(y)),
                                        feature_scale=float(s),
                                        feature_id=int(f1_id),
                                        feature_color=(float(r), float(g), float(b)))
                    # print("Adding edge {} for lm_id: {}: ".format(f1_id, lm_id))
                    # print("lm num_observed: ", lm.num_observed, lm.num_observable)

                
                pos_w = self.reconstruction.points[lm_id].coordinates
                lm.update_normal_and_depth(pos_w, self.graph)
                lm.compute_descriptor(self.curr_kf, self.graph)
                n_new_edges += 1
        print("Added ", n_new_edges, " for kf: ", self.curr_kf.im_name)
        #TODO: update graph connections
        #TODO: self.add_keyframe_to_map(self.curr_kf)
        # Is that necessary

    def assign_features_to_lms(self, new_kf: Keyframe):
        margin = 5
        # lk_landmarks = self.last_lk
        print("lk_landmarks: ", self.last_lk)
        print("landmarks: ", new_kf.landmarks_)

        curr_lms =  []
        for lm_id, _ in self.last_lk:
            curr_lms.append(lm_id)
        print("curr_lms: ", curr_lms)
        print("lens: ", len(self.last_lk), " lms: ", len(new_kf.landmarks_), " curr_lms: ", len(curr_lms))
        # merge seen landmarks with all landmarks in the local map
        # new_kf.landmarks_[:] = list(set(new_kf.landmarks_+self.local_landmarks))
        curr_lms[:] = list(set(curr_lms+self.local_landmarks))

        # check for duplicates
        lms = defaultdict(int)

        for lm_id in curr_lms:
            lms[lm_id] += 1
            if lms[lm_id] > 1:
                print("Double lm_id: ", lm_id)

        matches = slam_matcher.\
            match_frame_to_landmarks(new_kf.descriptors, curr_lms, margin,
                                     self.data, self.graph)
        
        # # debug: visualize matches
        p1, f1, c1 = new_kf.load_points_desc_colors()
        # print("self.curr_kf: ", self.curr_kf.im_name, " new_kf: ", new_kf.im_name)
        # p21, f21, _ = self.curr_kf.load_points_desc_colors()
        # p2 = []
        # for f_id, lm_idx in matches:
        #     lm_id = curr_lms[lm_idx]
        #     print("g:", lm_id, type(lm_id),",", self.graph[lm_id], ", ", self.graph.node[lm_id])
        #     p2.append(self.graph.get_edge_data(self.curr_kf.im_name, lm_id)['feature'])
        # p2 = np.asarray(p2)
        # slam_debug.disable_debug = False
        # slam_debug.visualize_matches_pts(p1,p2,np.column_stack((matches[:,0],np.arange(0, len(p2)))), self.data.load_image(new_kf.im_name), self.data.load_image(self.init_frame.im_name), True)
        # slam_debug.disable_debug = True

        # matches2 = slam_matcher.match_desc_and_points(self.data, f1, f21, p1, p21, self.camera[1])
        # # TODO: remove debug
        # print("len(matches): ", len(matches), len(matches2))
        # Let's assume that matches are mostly correct and matched landmarks are visible!
        if matches is None:
            return
        n_added = 0
        self.feature_ids_last_frame = {}
        for f1_id, lm_idx in matches:
            lm_id = curr_lms[lm_idx]
            self.feature_ids_last_frame[lm_id] = f1_id
            lm = self.graph.node[lm_id]['data']
            lm.num_observable += 1
            if self.graph.has_node(lm_id):
                print("Creating edge: ", new_kf.im_name, lm_id, " f1_id: ", f1_id)
                x, y, s = p1[f1_id, 0:3]
                r, g, b = c1[f1_id, :]
                # add observations
                self.graph.add_edge(new_kf.im_name, lm_id,
                                    feature=(float(x), float(y)),
                                    feature_scale=float(s),
                                    feature_id=int(f1_id),
                                    feature_color=(float(r), float(g), float(b)))
                pos_w = self.reconstruction.points[lm_id].coordinates
                lm.update_normal_and_depth(pos_w, self.graph)
                lm.compute_descriptor(new_kf, self.graph)
                new_kf.matched_lms[f1_id] = lm_id
                n_added += 1

        print("matches: ", len(matches), " add obs: ", n_added, " new: ", len(matches)-n_added)
        return

    def add_fresh_landmark(self, lm_id):
        # return
        self.fresh_landmarks.append(lm_id)

    def create_new_keyframe(self, frame):
        self.fuse_duplicated_landmarks()
        self.update_local_map_lk(frame)
        self.fuse_duplicated_landmarks()
        pdc = slam_utils.extract_features(frame.im_name, self.data)
        new_kf = Keyframe(frame, self.data, self.n_keyframes, pdc)
        print("self.local_kf: ", self.local_keyframes)
        self.add_keyframe(new_kf)
        self.fuse_duplicated_landmarks()
        self.assign_features_to_lms(new_kf)
        self.fuse_duplicated_landmarks()
        self.mapping_with_new_keyframe_lk(new_kf)
        self.fuse_duplicated_landmarks()
        self.local_bundle_adjustment2()
        self.fuse_duplicated_landmarks()
        self.update_lk_landmarks()
        self.fuse_duplicated_landmarks()

    # OpenVSlam optimize_current_frame_with_local_map
    def track_with_local_map(self, frame: Frame, slam_tracker):
        """Refine the pose of the current frame with the "local" KFs"""
        print("track_with_local_map", len(frame.landmarks_))
        # acquire more 2D-3D matches by reprojecting the local landmarks to the current frame
        matches = self.search_local_landmarks(frame)
        matches = np.array(matches)
        print("track_with_local_map: matches: ", len(matches))
        observations, _, _ = frame.load_points_desc_colors()
        print("load_features: ", len(observations))
        print("observations.shape: ", np.shape(observations), matches[:, 0].shape)
        observations = observations[matches[:, 0], 0:3]
        print("len(observations): ", len(observations), observations.shape,
              len(self.local_landmarks))

        points3D = np.zeros((len(observations), 3))

        print("self.reconstruction: ", len(self.reconstruction.points),
              len(points3D), len(frame.landmarks_), len(matches))

        for (pt_id, (m1, m2)) in enumerate(matches):
            lm_id = frame.landmarks_[m2] 
            points3D[pt_id, :] = \
                self.reconstruction.points[str(lm_id)].coordinates


        print("points3D.shape: ", points3D.shape,
              "observations.shape: ", observations.shape)
        print("frame.world_pose: ", frame.im_name,
              frame.world_pose.rotation, frame.world_pose.translation)
        #TODO: Remove debug stuff
        slam_debug.reproject_landmarks(points3D, observations, frame.world_pose, 
                                       frame.im_name, self.camera[1], self.data,
                                       title="bef tracking: "+frame.im_name, do_show=False)
        pose, valid_pts = slam_tracker.\
            bundle_tracking(points3D, observations,
                            frame.world_pose, self.camera,
                            self.data.config, self.data)
        
        print("pose after! ", pose.rotation, pose.translation)
        print("valid_pts: ", len(valid_pts), " vs ", len(observations))
        slam_debug.reproject_landmarks(points3D, observations,
                                       pose, frame.im_name, self.camera[1], self.data,
                                       title="aft tracking: "+frame.im_name, do_show=True)
        
        frame.update_visible_landmarks(matches[:, 1])
        frame.landmarks_ = list(compress(frame.landmarks_, valid_pts))
        self.num_tracked_lms = len(frame.landmarks_)
        frame.world_pose = pose
        m = matches[:, 0][valid_pts]
        self.feature_ids_last_frame = {}
        # TODO: avoid double load
        observations, _, _ = frame.load_points_desc_colors()
        # add observations but do not add edge!
        for idx, lm_id in enumerate(frame.landmarks_):
            m1 = m[idx]
            self.feature_ids_last_frame[lm_id] = m1
            lm = self.graph.nodes[lm_id]['data']
            lm.num_observed += 1
        return pose

    def new_kf_needed(self, frame: Frame):
        """Return true if a new keyframe is needed based on the OpenVSLAM criteria
        """
        if len(self.keyframes) == 2: # just initialized
            print("NEW KF", frame.im_name)
            return True
        print("self.n_keyframes: ", self.n_keyframes)
        # Count the number of 3D points observed from more than 3 viewpoints
        min_obs_thr = 3 if 3 <= self.n_keyframes else 2

        # #essentially the graph
        # #find the graph connections
        # #it's about the observations in all frames and not just the kfs
        # #so we can't use the graph of only kfs
        # num_reliable_lms = get_tracked_landmarks(min_obs_thr)
        num_reliable_lms = self.curr_kf.\
            get_num_tracked_landmarks(min_obs_thr, self.graph)
        
        if len(self.graph[self.curr_kf.im_name])*0.75 > len(self.last_lk) and self.config_slam['tracker_lk']:
            print("New KF criteria!")
            return True
        print("num_reliable_lms: ", num_reliable_lms)
        max_num_frms_ = 30  # the fps
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
        # # Do not add if B is not satisfied
        if not cond_b:
            print("not cond_b -> no kf")
            return False
        
        # # Do not add if none of A is satisfied
        if not cond_a1 and not cond_a2 and not cond_a3:
            print("not cond_a1 and not cond_a2 and not cond_a3 -> no kf")
            return False
        print("NEW KF", frame.im_name)
        # exit()
        return True

    # def triangulate_with_two_kfs_old(self, kf1: Keyframe, kf2: Keyframe, matches, data):

    #     """kf1 -> neighbor (old) kf, kf2 -> current kf
    #     """
    #     #load the features to be triangulated
    #     #TODO: Think about frame1/2 and matches
    #     frame1, frame2 = kf1.im_name, kf2.im_name
    #     p1, f1, c1 = feature_loader.instance.load_points_features_colors(
    #                  data, frame1, masked=True)
    #     p2, f2, c2 = feature_loader.instance.load_points_features_colors(
    #                  data, frame2, masked=True)

    #     slam_debug.visualize_matches(matches, frame1, frame2, data, True)
    #     # Maybe we have double matches
    #     d_m1 = defaultdict(int)
    #     d_m2 = defaultdict(int)
    #     for (m1, m2) in matches:
    #         d_m1[m1] += 1
    #         d_m2[m2] += 1
    #         if d_m1[m1] > 1 or d_m2[m2] > 1:
    #             print("Double matches!!", m1, m2)
    #             exit()
    #     in_graph = {}
    #     # seen_landmarks = self.graph[frame2]
    #     seen_landmarks = self.graph[frame1]
    #     print("frame1: ", frame1, " frame2: ", frame2)
    #     for lm_id in seen_landmarks:
    #         e = self.graph.get_edge_data(frame1, lm_id)
    #         if e is None:
    #             continue
    #         if e['feature_id'] in in_graph:
    #             e2 = self.graph.get_edge_data(frame2, lm_id)
    #             print("e(", frame1, ",", lm_id, "): ", e)
    #             print("e2(", frame2, ",", lm_id, "): ", e2)
    #             print("Already in there first!", e['feature_id'],
    #                   "lm_id: ", lm_id)
    #             exit()
    #         in_graph[e['feature_id']] = lm_id

    #     print("len(in_graph)", len(in_graph),
    #           "frames: {} {}".format(frame1, frame2))
    #     added_obs = np.ones(len(matches), dtype=bool)
    #     n_added = 0
    #     for (idx, (m1, m2)) in enumerate(matches):
    #         # if the feature id is matched a is not none
    #         a = in_graph.get(m1)
    #         #If it is a match, just add the observations
    #         #TODO: don't re-add stuff
    #         print("Testing m1 {} and m2 {} ".format(m1, m2))
    #         if a is None:
    #             x, y, s = p1[m1, 0:3]
    #             r, g, b = c1[m1, :]
    #             print("----")
    #             print("data1: ", frame1,
    #                   self.graph.get_edge_data(str(a), str(frame1)))
    #             print("data2: ", frame2,
    #                   self.graph.get_edge_data(str(a), str(frame2)))
    #             print("adding: ", frame1, "lm_id", a, "x,y,s", x, y, s, 
    #                   "r,g,b: ", r, g, b, "m1", m1)
    #             added_obs[idx] = True
    #             n_added += 1
    #         else:
    #             print("Filtering m1, m2: ", m1, m2)
    #             print("m1 {} already in there".format(m1))
    #             added_obs[idx] = False

    #     matches = matches[added_obs, :]
    #     print("Remove already added observations: ",
    #           len(matches), len(added_obs))

    #     # match
    #     print("len(p1): {}, len(p2): {} ".format(len(p1), len(p2)))
    #     # Now, build up the graph for the triangulation

    #     # create the graph
    #     tracks_graph = nx.Graph()
    #     tracks_graph.add_node(str(frame1), bipartite=0)
    #     tracks_graph.add_node(str(frame2), bipartite=0)

    #     for (track_id, (f1_id, f2_id)) in enumerate(matches):
    #         print("track_id {}, frames: {}<->{} f1_id {}, f2_id {}".
    #               format(track_id, frame1, frame2, f1_id, f2_id))
    #         x, y, s = p2[f2_id, 0:3]
    #         if np.isnan(x):
    #             continue
    #         r, g, b = c2[f2_id, :]
    #         tracks_graph.add_node(str(track_id), bipartite=1)
    #         tracks_graph.add_edge(str(frame2),
    #                               str(track_id),
    #                               feature=(float(x), float(y)),
    #                               feature_scale=float(s),
    #                               feature_id=int(f2_id),
    #                               feature_color=(float(r), float(g), float(b)))

    #         x, y, s = p1[f1_id, 0:3]
    #         r, g, b = c1[f1_id, :]
    #         tracks_graph.add_edge(str(frame1),
    #                               str(track_id),
    #                               feature=(float(x), float(y)),
    #                               feature_scale=float(s),
    #                               feature_id=int(f1_id),
    #                               feature_color=(float(r), float(g), float(b)))

    #     cameras = data.load_camera_models()
    #     camera = next(iter(cameras.values()))
    #     rec_tri = types.Reconstruction()
    #     rec_tri.reference = data.load_reference()
    #     rec_tri.cameras = cameras

    #     shot1 = types.Shot()
    #     shot1.id = frame1
    #     shot1.camera = camera
    #     shot1.pose = kf1.world_pose
    #     shot1.metadata = reconstruction.get_image_metadata(data, frame1)
    #     rec_tri.add_shot(shot1)

    #     shot2 = types.Shot()
    #     shot2.id = frame2
    #     shot2.camera = camera
    #     shot2.pose = kf2.world_pose
    #     shot2.metadata = reconstruction.get_image_metadata(data, frame2)
    #     rec_tri.add_shot(shot2)

    #     graph_inliers = nx.Graph()

    #     print("Running triangulate shot features for ", frame2)
    #     np_before = len(rec_tri.points)
    #     reconstruction.triangulate_shot_features(tracks_graph, graph_inliers,
    #                                              rec_tri, frame1,
    #                                              data.config)
    #     np_after = len(rec_tri.points)
    #     print("Created len(graph_inliers.nodes()): ",
    #           len(graph_inliers.nodes()))
    #     print("Points before: {} and {} ".format(np_before, np_after))
    #     # visualize landmarks 2D points in KF <-> 2D points in new KF
    #     # and also reprojections!
    #     # draw triangulate features in im1
    #     # get observations
    #     edges1 = graph_inliers.edges(frame1)
    #     edges2 = graph_inliers.edges(frame2)
    #     # we have the edges
    #     # try to find the same feature already existing in the graph!
    #     n_duplicates = 0
    #     for u, v in edges1:
    #         feature_id = graph_inliers.get_edge_data(u, v)['feature_id']
    #         for lm_id in self.graph[frame1]:
    #             feature_id2 = self.graph.\
    #                 get_edge_data(frame1, lm_id)['feature_id']
    #             if feature_id == feature_id2:
    #                 print("created feature already in graph",
    #                       feature_id, "<->", feature_id2)
    #                 print("u,v", u, v)
    #                 print("frame1", frame1, "lm_id", lm_id)
    #                 print(self.graph[lm_id])
    #                 n_duplicates += 1
    #                 exit()

    #     print("Created landmarks ", np_after, " with ",
    #           n_duplicates, " duplicates.")
    #     logger.setLevel(logging.INFO)
    #     points = rec_tri.points
    #     points3D = np.zeros((len(points), 3))
    #     for idx, pt3D in enumerate(points.values()):
    #         points3D[idx, :] = pt3D.coordinates
    #     DO_VISUALIZE = False
    #     if DO_VISUALIZE:
    #         obs1 = []
    #         for u, v in edges1:
    #             obs1.append(graph_inliers.get_edge_data(u, v)['feature'])
    #         print("obs1: ", obs1)
    #         slam_debug.draw_observations_in_image(np.asarray(obs1), frame1, data, False)
    #         obs2 = []
    #         for u, v in edges2:
    #             obs2.append(graph_inliers.get_edge_data(u, v)['feature'])
    #         print("obs2: ", obs2)
    #         slam_debug.draw_observations_in_image(np.asarray(obs2), frame2, data, False)
    #         logger.setLevel(logging.INFO)

    #     # Due to some sorting issues, we have to go through
    #     # graph_inliers by "frames" first
    #     for _, gi_lm_id in graph_inliers.edges(frame1):
    #         lm_id = str(self.current_lm_id)
    #         lm = Landmark(lm_id)
    #         self.n_landmarks += 1
    #         self.current_lm_id += 1
    #         # This is essentially the same as adding it to the graph
    #         self.add_landmark(lm)
    #         # TODO: observations
    #         self.graph.add_edges_from([(frame1, str(lm_id), graph_inliers.
    #                                     get_edge_data(frame1, gi_lm_id))])
    #         self.graph.add_edges_from([(frame2, str(lm_id), graph_inliers.
    #                                     get_edge_data(frame2, gi_lm_id))])
            
    #         print("graph_inliers.get_edge_data(",frame1, "gi_lm_id): ", graph_inliers.
    #                                     get_edge_data(frame1, gi_lm_id), "lm_id: ", lm_id)
    #         print("graph_inliers.get_edge_data(", frame2, "gi_lm_id): ", graph_inliers.
    #                                     get_edge_data(frame2, gi_lm_id), "lm_id: ", lm_id)
    #         lm.compute_descriptor(self.graph)
    #         lm.update_normal_and_depth(pt3D.coordinates, self.graph)
    #         # We also have to add the points to the reconstruction
    #         point = types.Point()
    #         point.id = str(lm_id)
    #         point.coordinates = rec_tri.points[gi_lm_id].coordinates
    #         self.reconstruction.add_point(point)
    #         self.local_landmarks.append(lm.lm_id)

    #     points3D_debug = np.zeros([len(rec_tri.points),3])
    #     for idx, p in enumerate(rec_tri.points.values()):
    #         points3D_debug[idx, :] = p.coordinates
    #     print("Visualizing ")
    #     slam_debug.reproject_landmarks(points3D_debug, np.zeros([len(rec_tri.points),2]),
    #                                    kf2.world_pose, kf2.im_name, camera, self.data, 
    #                                    title="triangulated: "+kf2.im_name, do_show=True)        

    # def add_frame_to_reconstruction(self, frame, pose, camera, data):
    #     shot1 = types.Shot()
    #     shot1.id = frame
    #     print("add_frame_to_reconstructioncamera: ", camera)
    #     print("add_frame_to_reconstructioncamera: ", camera[1].id)
    #     print("add_frame_to_reconstruction frame: ", frame)
    #     shot1.camera = camera[1]
    #     shot1.pose = types.Pose(pose.rotation, pose.translation)
    #     shot1.metadata = reconstruction.get_image_metadata(data, frame)
    #     self.reconstruction.add_shot(shot1)

    # def set_curr_kf(self, keyframe):
    #     """Sets a new keyframe

    #     Arguments:
    #         keyframe: of type Frame
    #     """
    #     self.curr_kf = keyframe
    #     self.n_keyframes += 1
    #     self.set_last_frame(keyframe)
    #     # TODO: Think about initializing the new keyframe with the
    #     #       old landmarks
    #     self.keyframes[keyframe.id] = keyframe

        # def estimate_pose(self):
    #     # TODO: Implement constant velocity model
    #     return self.last_frame.world_pose