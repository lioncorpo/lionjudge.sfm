from opensfm import reconstruction
from opensfm import matching
from opensfm import types
from opensfm import feature_loader
from opensfm import features
import slam_utils
import slam_debug
import cslam
import numpy as np
import logging
import networkx as nx
logger = logging.getLogger(__name__)


class SlamInitializer(object):

    def __init__(self, data, camera, guided_matcher):
        print("initializer")
        self.init_type = "OpenVSlam"
        self.init_frame = None
        self.prev_pts = None
        self.data = data
        self.camera = camera
        self.guided_matcher = guided_matcher

    def set_initial_frame(self, frame):
        """Sets the first frame"""
        self.init_frame = frame
        self.prev_pts = frame.cframe.getKptsUndist()[:, 0:2]

    def initialize_openvslam(self, frame):
        """Initialize similar to ORB-SLAM and Openvslam"""
        print("initialize_openvslam")
        # We should have two frames: the current one and the init frame
        
        # TODO: think about prev_matches!
        matches = self.guided_matcher.\
            match_frame_to_frame(self.init_frame.cframe,
                                 frame.cframe, self.prev_pts,
                                 100)
        matches = np.array(matches)
        print("Init: ", len(matches))
        # Update pts
        self.prev_pts[matches[0, :], :] =\
            frame.cframe.getKptsUndist()[matches[1, :], 0:2]

        f1_points = self.init_frame.cframe.getKptsPy()
        f2_points = frame.cframe.getKptsPy()
        
        # test reconstructability
        threshold = 4 * self.data.config['five_point_algo_threshold']
        args = []
        im1 = self.init_frame.im_name
        im2 = frame.im_name
        norm_p1 = features.\
            normalized_image_coordinates(f1_points[matches[:, 0], 0:2], self.camera[1].width, self.camera[1].height)
        norm_p2 = features.\
            normalized_image_coordinates(f2_points[matches[:, 1], 0:2], self.camera[1].width, self.camera[1].height)
        norm_size = max(self.camera[1].width, self.camera[1].height)

        # slam_debug.visualize_matches_pts(norm_p1, norm_p2, np.column_stack((np.arange(0,len(norm_p1)), np.arange(0,len(norm_p1)))),
        #                                  self.init_frame.image,
        #                                  frame.image, is_normalized=True, do_show=True)
        args.append((im1, im2, norm_p1, norm_p2,
                     self.camera[1], self.camera[1], threshold))
        chrono = reconstruction.Chronometer()
        chrono.lap("others")
        i1, i2, r = reconstruction._compute_pair_reconstructability(args[0])
        chrono.lap("pair rec")
        if r == 0:
            return None, None, None
        scale_1 = f1_points[matches[:, 0], 2] / norm_size
        scale_2 = f2_points[matches[:, 1], 2] / norm_size
        # create the graph
        tracks_graph = nx.Graph()
        tracks_graph.add_node(str(im1), bipartite=0)
        tracks_graph.add_node(str(im2), bipartite=0)
        # p1, p2 = self.init_frame.points[mat, 0:2]
        for (track_id, (f1_id, f2_id)) in enumerate(matches):
            x, y = norm_p1[track_id, 0:2]
            s = scale_1[track_id]
            r, g, b = self.init_frame.colors[f1_id, :]
            tracks_graph.add_node(str(track_id), bipartite=1)
            tracks_graph.add_edge(str(im1),
                                  str(track_id),
                                  feature=(float(x), float(y)),
                                  feature_scale=float(s),
                                  feature_id=int(f1_id),
                                  feature_color=(float(r), float(g), float(b)))
            x, y = norm_p2[track_id, 0:2]
            s = scale_2[track_id]
            r, g, b = frame.colors[f2_id, 0:3]
            tracks_graph.add_edge(str(im2),
                                  str(track_id),
                                  feature=(float(x), float(y)),
                                  feature_scale=float(s),
                                  feature_id=int(f2_id),
                                  feature_color=(float(r), float(g), float(b)))
        chrono.lap("track graph")
        rec_report = {}
        reconstruction_init, graph_inliers, rec_report['bootstrap'] = \
            reconstruction.bootstrap_reconstruction(self.data, tracks_graph, self.data.load_camera_models(),
                                                    im1, im2, norm_p1, norm_p2)
        chrono.lap("boot rec")
        print("Init timings: ", chrono.lap_times())
        if reconstruction_init is not None:
            print("Created init rec from {}<->{} with {} points from {} matches"
                  .format(im1, im2, len(reconstruction_init.points), len(matches)))
        slam_debug.visualize_graph(graph_inliers, self.init_frame.im_name, frame.im_name, self.data, do_show=True)
        return reconstruction_init, graph_inliers, matches
        


    def initialize_iccv(self, data, frame):
        """Initialize similar Elaborate Monocular Point and Line SLAM 
            With Robust Initialization
        """
        print("initialize_openvslam")

    def initialize(self, frame): #data, config_slam, frame):
        # if self.init_type == "ICCV":
            # return self.initialize_iccv(data, frame)
        # if self.init_type == "OpenSfM":
            # return self.initialize_opensfm(data, config_slam, frame)
        return self.initialize_openvslam(frame)


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