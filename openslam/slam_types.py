from opensfm import types
import logging
import numpy as np
import slam_utils
from opensfm import feature_loader
logger = logging.getLogger(__name__)


class Landmark(object):

    def __init__(self, graph_id):
        """Creates a landmark

        Arguments:
            graph_id : The id the landmark has in the graph (must be unique)
        """
        self.lm_id = graph_id
        self.is_observable_in_tracking = False
        self.local_map_update_identifier = -1  # the last frame where it was observed
        self.identifier_in_local_lm_search_ = -1
        self.n_observable = 0  # the number of frames and KFs it is seen in
        self.descriptor = None
        self.num_observable = 1  # landmarks visible in current frame (maybe without match)
        self.num_observed = 1  # landmarks matched in current frame
        self.first_kf_id = -1
        self.ref_kf = None
        self.mean_normal = []

    def get_observed_ratio(self):
        return self.num_observed/self.num_observable

    def update_normal_and_depth(self, pos_w, graph):
        observations = graph[str(self.lm_id)]
        if len(observations) == 0:
            return
        self.mean_normal = np.array([0., 0., 0.])
        for kf_name in observations.keys():
            kf = graph.nodes[kf_name]['data']
            normal = pos_w - kf.world_pose.get_origin()
            self.mean_normal += (normal / np.linalg.norm(normal))

        # n_observations = len(self.observations)
        # cam_to_lm_vec = self.pos_w - self.ref_kf.pose.world_pose()
        # dist = cam_to_lm_vec.norm()
        #TODO: scale level, scale factor, depth
        self.mean_normal /= len(observations)

    # def add_observation(self, kf, idx):
    #     """ Adds an observation

    #     An observation is defined by the kf and the id of the feature in this kf
    #     """
    #     if kf in self.observations:
    #         return
    #     self.observations[kf] = idx
    #     self.num_observations += 1

    # def compute_descriptor(self, graph):
    #     """ Computes the descriptor from the observations
    #     - similar to OpenVSlam
    #     - or simply take the most recent one
    #     """
    #     """Computes the descriptor of the lm
    #     from all the observations
    #     Take random descriptor
    #     """

    #     keyframes = graph[str(self.lm_id)]
    #     print("keyframes: ", keyframes)
    #     # for kf_name in keyframes:
    #     #     kf: Keyframe = graph.nodes[kf_name]['data']
    #     #     track = graph.get_edge_data(kf_name, str(self.lm_id))
    #     #     self.descriptor = kf.descriptors[track['feature_id']]
    #     #     return
    #             # for kf_name in keyframes:
    #     kf_name = keyframes[-1]
    #     kf: Keyframe = graph.nodes[kf_name]['data']
    #     track = graph.get_edge_data(kf_name, str(self.lm_id))
    #     self.descriptor = kf.descriptors[track['feature_id']]
        # return

    def compute_descriptor(self, kf, graph):
        """ Computes the descriptor from the observations
        - similar to OpenVSlam
        - or simply take the most recent one
        """
        """Computes the descriptor of the lm
        from all the observations
        Take random descriptor
        """

        # keyframes = graph[str(self.lm_id)]
        # print("keyframes: ", keyframes)
        # for kf_name in keyframes:
        #     kf: Keyframe = graph.nodes[kf_name]['data']
        #     track = graph.get_edge_data(kf_name, str(self.lm_id))
        #     self.descriptor = kf.descriptors[track['feature_id']]
        #     return
                # for kf_name in keyframes:
        # kf_name = keyframes[-1]
        # kf: Keyframe = graph.nodes[kf_name]['data']
        track = graph.get_edge_data(kf.im_name, str(self.lm_id))
        self.descriptor = kf.descriptors[track['feature_id']]
        # return


class Frame(object):
    def __init__(self, name, id):
        print("Creating frame: ", name)
        self.im_name = name
        self.landmarks_ = []
        self.idx_valid = None
        self.frame_id = id
        self.kf_id = -1  # if non-KF, id of "parent" KF
        self.is_keyframe = False
        self.world_pose = types.Pose()
        self.rel_pose_to_kf = types.Pose()
        #stores where the frame was last updated
        self.local_map_update_identifier = -1
        self.lk_pyramid = None

        self.has_features = False
        self.descriptors = None
        self.points = None
        self.colors = None

    def load_points_desc_colors(self):
        if self.has_features:
            return (self.points, self.descriptors, self.colors)
        return None

    def extract_features(self, data, do_extract=False):
        """Loads or extracts descriptors, points and colors"""
        if do_extract:
            self.points, self.descriptors, self.colors =\
                slam_utils.extract_features(self.im_name, data)
        else:
            self.points, self.descriptors, self.colors = \
                feature_loader.instance.load_points_features_colors(
                    data, self.im_name, masked=True)

        self.has_features = (self.points is not None and
                             self.descriptors is not None and
                             self.colors is not None)

    def update_visible_landmarks_old(self, idx):
        if self.visible_landmarks is None:
            return
        self.visible_landmarks = self.visible_landmarks[idx, :]

    def update_visible_landmarks(self, idx):
        print("before landmarks: ", len(self.landmarks_))
        self.landmarks_[:] = [self.landmarks_[m1] for m1 in idx]
        print("after landmarks: ", len(self.landmarks_))

    def set_visible_landmarks(self, points, inliers):
        self.visible_landmarks = points  # id, coordinates
        self.idx_valid = np.zeros(len(inliers.values()))
        for (idx, feature) in enumerate(inliers.values()):
            self.idx_valid[idx] = feature['feature_id']

    def store(self):
        """Reduces the object to just the header"""
        self.visible_landmarks = []


class Keyframe(object):
    def __init__(self, frame: Frame, data, kf_id, pdc=None):
        """pdc is points, descriptors and colors as "triple"
        """
        # The landmarks store the id of the lms in the graph
        self.landmarks_ = frame.landmarks_.copy()
        print("Creating KF: ", kf_id, len(self.landmarks_), frame.im_name)
        self.im_name = frame.im_name  # im_name should also be unique
        self.kf_id = kf_id  # unique_id
        self.frame_id = frame.frame_id
        # check if features already exist in the frame
        if pdc is not None:
            self.points, self.descriptors, self.colors = pdc
        else:
            self.points, self.descriptors, self.colors = \
                feature_loader.instance.load_points_features_colors(
                    data, self.im_name, masked=True)
        self.matched_lms = np.ones(len(self.descriptors), dtype=int)*-1
        self.world_pose = frame.world_pose
        self.local_map_update_identifier = -1
        
    def load_points_desc_colors(self):
        return (self.points, self.descriptors, self.colors)

    # def add_landmark(self, lm: Landmark):
    #     self.landmarks_[lm.lm_id] = lm

    def get_num_tracked_landmarks(self, min_obs_thr, graph):
        """Counts the number of reliable landmarks, i.e. all visible in
        greater or equal `min_obs_thr` keyframes
        """
        print("get_num_tracked_landmarks: ", self.kf_id, self.frame_id)
        print("min_obs_thr: ", min_obs_thr)
        print("tracked: ", len(self.landmarks_))
        if min_obs_thr > 0:
            n_lms = 0
            for lm_id in graph[self.im_name]:
                if len(graph[lm_id]) >= min_obs_thr:
                    n_lms += 1
            print("n_lms: ", n_lms, ", ", len(graph[self.im_name]))
            return n_lms
        return len(self.visible_landmarks)

    def compute_median_depth(self, absval, graph, reconstruction):
        Rt = self.world_pose.get_Rt()
        rot_cw_z_row = Rt[2, 0:3]
        trans_cw_z = Rt[2, 3]
        depths = []
        print("kf_id: ", self.kf_id)
        for lm_id in self.landmarks_:
            if graph.has_node(str(lm_id)):
                pos_w = reconstruction.points[lm_id].coordinates
                pos_c_z = np.dot(rot_cw_z_row, pos_w) + trans_cw_z
                depths.append(pos_c_z)

        if len(depths) == 0:
            return -1

        if absval:
            return np.median(np.abs(depths))
        return np.median(depths)
