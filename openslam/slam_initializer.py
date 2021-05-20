from opensfm import reconstruction
from opensfm import matching
from opensfm import types
from opensfm import feature_loader
from opensfm import features
import slam_utils
import slam_debug
import slam_matcher
import numpy as np
import logging
import networkx as nx
logger = logging.getLogger(__name__)


class SlamInitializer(object):

    def __init__(self, config): #, slam_matcher):
        print("initializer")
        self.init_type = "OpenSfM"
        self.init_frame = None
        self.init_pdc = None # points, descriptors and colors of init frame

    def set_initial_frame(self, data, frame):
        """Sets the first frame"""
        self.init_frame = frame
        if frame.has_features:
            self.init_pdc = frame.load_points_desc_colors()

    def initialize_opensfm(self, data, config_slam, frame):
        chrono = reconstruction.Chronometer()
        im1, im2 = self.init_frame.im_name, frame.im_name

        if frame.has_features:
            self.other_pdc = frame.load_points_desc_colors()
        else:
            self.other_pdc = slam_utils.extract_features(im2, data)
        p1, f1, c1 = self.init_pdc
        p2, f2, c2 = self.other_pdc
        chrono.lap("loading p,f,c")
        threshold = data.config['five_point_algo_threshold']
        cameras = data.load_camera_models()
        camera = next(iter(cameras.values()))
        # success, matches = slam_matcher.match(data, im1, im2, camera)
        success, matches = slam_matcher.\
            match_desc_and_points(data, f1, f2, p1, p2, camera)
        chrono.lap("matching")
        if not success:
            return None, None, None
        p1, p2 = p1[matches[:, 0], :], p2[matches[:, 1], :]
        f1, f2 = f1[matches[:, 0], :], f2[matches[:, 1], :]
        c1, c2 = c1[matches[:, 0], :], c2[matches[:, 1], :]

        threshold = 4 * data.config['five_point_algo_threshold']
        args = []
        args.append((im1, im2, p1[:, 0:2], p2[:, 0:2],
                     camera, camera, threshold))
        chrono.lap("others")
        i1, i2, r = reconstruction._compute_pair_reconstructability(args[0])
        chrono.lap("pair rec")
        if r == 0:
            return None, None, None
        # create the graph
        tracks_graph = nx.Graph()
        tracks_graph.add_node(str(im1), bipartite=0)
        tracks_graph.add_node(str(im2), bipartite=0)
        for (track_id, (f1_id, f2_id)) in enumerate(matches):
            x, y, s = p1[track_id, 0:3]
            r, g, b = c1[track_id, :]
            tracks_graph.add_node(str(track_id), bipartite=1)
            tracks_graph.add_edge(str(im1),
                                  str(track_id),
                                  feature=(float(x), float(y)),
                                  feature_scale=float(s),
                                  feature_id=int(f1_id),
                                  feature_color=(float(r), float(g), float(b)))
            x, y, s = p2[track_id, 0:3]
            r, g, b = c2[track_id, :]
            tracks_graph.add_edge(str(im2),
                                  str(track_id),
                                  feature=(float(x), float(y)),
                                  feature_scale=float(s),
                                  feature_id=int(f2_id),
                                  feature_color=(float(r), float(g), float(b)))
        chrono.lap("track graph")
        rec_report = {}
        reconstruction_init, graph_inliers, rec_report['bootstrap'] = \
            reconstruction.bootstrap_reconstruction(data, tracks_graph,
                                                    im1, im2, p1[:, 0:2], p2[:, 0:2])
        chrono.lap("boot rec")
        print("Init timings: ", chrono.lap_times())
        print("Created init rec from {}<->{} with {} points from {} matches"
              .format(im1, im2, len(reconstruction_init.points), len(matches)))
        return reconstruction_init, graph_inliers, matches

    def initialize_openvslam(self, data, frame):
        """Initialize similar to ORB-SLAM and Openvslam"""
        print("initialize_openvslam")

    def initialize_iccv(self, data, frame):
        """Initialize similar Elaborate Monocular Point and Line SLAM 
            With Robust Initialization
        """
        print("initialize_openvslam")

    def initialize(self, data, config_slam, frame):
        if self.init_type == "ICCV":
            return self.initialize_iccv(data, frame)
        if self.init_type == "OpenSfM":
            return self.initialize_opensfm(data, config_slam, frame)
        return self.initialize_openvslam(data, frame)
