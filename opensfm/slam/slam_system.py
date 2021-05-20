from opensfm import pyslam
from opensfm import pymap
from opensfm import dataset
from slam_initializer import SlamInitializer
import numpy as np
import slam_config
import slam_debug
import slam_utils
from slam_mapper import SlamMapper
from slam_tracker import SlamTracker
import logging
logger = logging.getLogger(__name__)


class SlamSystem(object):

    def __init__(self, args):
        self.data = dataset.DataSet(args.dataset)
        self.config = self.data.config
        self.config_slam = slam_config.default_config()
        self.camera = next(iter(self.data.load_camera_models().items()))
        self.map = pymap.Map()

        # Create the camera model
        c = self.camera[1]
        self.cam = pymap.BrownPerspectiveCamera(
            c.width, c.height, c.projection_type,
            c.focal_x, c.focal_y, c.c_x, c.c_y, c.k1, c.k2, c.p1, c.p2, c.k3
        )
        # Create the matching shot camera
        self.shot_cam = self.map.create_shot_camera(0, self.cam, self.camera[0])

        self.extractor = pyslam.OrbExtractor(
            self.config_slam['feat_max_number'],
            self.config_slam['feat_scale'],
            self.config_slam['feat_pyr_levels'],
            self.config_slam['feat_fast_ini_th'],
            self.config_slam['feat_fast_min_th']
        )

        corner_pts = np.array([[0, 0],  # left top
                               [self.camera[1].width, 0],  # right top
                               [0, self.camera[1].height],  # left bottom
                               [self.camera[1].width, self.camera[1].height]])  # right bottom

        corners = self.camera[1].undistort_many(corner_pts).reshape((4, 2))
        bounds = np.array([np.min((corners[0, 0], corners[2, 0])),
                           np.max((corners[1, 0], corners[3, 0])),
                           np.min((corners[0, 1], corners[2, 1])),
                           np.max((corners[1, 1], corners[3, 1]))])
        inv_cell_w = self.config_slam['grid_n_cols'] / (bounds[1] - bounds[0])
        inv_cell_h = self.config_slam['grid_n_rows'] / (bounds[3] - bounds[2])
        self.grid_params =\
            pyslam.\
            GridParameters(self.config_slam['grid_n_cols'],
                           self.config_slam['grid_n_rows'],
                           bounds[0], bounds[2], bounds[1], bounds[3],
                           inv_cell_w, inv_cell_h)
        self.matcher = pyslam.GuidedMatcher(self.grid_params,
                                            self.config_slam['feat_scale'],
                                            self.config_slam['feat_pyr_levels'])
        self.slam_mapper = SlamMapper(
            self.data, self.config_slam, self.camera, self.map, self.extractor, self.matcher)
        self.slam_init =\
            SlamInitializer(self.data, self.camera, self.matcher)
        self.slam_tracker = SlamTracker(self.matcher)
        self.system_initialized = False
    
    def process_frame(self, im_name, gray_scale_img):
        shot_id = self.map.next_unique_shot_id()
        curr_shot: pymap.Shot = self.map.create_shot(
            shot_id, self.shot_cam, im_name)
        print("Created shot: ", curr_shot.name, curr_shot.id)
        self.extractor.extract_to_shot(curr_shot, gray_scale_img, np.array([]))
        print("Extracted: ", curr_shot.number_of_keypoints())
        curr_shot.undistort_keypts()
        curr_shot.undistorted_keypts_to_bearings()
        self.matcher.distribute_undist_keypts_to_grid(curr_shot)
        chrono = slam_debug.Chronometer()
        if not self.system_initialized:
            self.system_initialized = self.init_slam_system(curr_shot)
            chrono.lap("init_slam_system_all")
            return self.system_initialized

        # Tracking
        pose = self.track_frame(curr_shot)
        chrono.lap("track")
        if pose is not None:
            curr_shot.get_pose().set_from_world_to_cam(slam_utils.pose_to_mat(pose))

        self.slam_mapper.update_with_last_frame(curr_shot)
        self.slam_mapper.num_tracked_lms = self.slam_tracker.num_tracked_lms
        if self.slam_mapper.new_keyframe_is_needed(curr_shot):
            self.slam_mapper.insert_new_keyframe(curr_shot)
        slam_debug.avg_timings.addTimes(chrono.laps_dict)
        return pose is not None

    def init_slam_system(self, curr_shot: pymap.Shot):
        """Find the initial depth estimates for the slam map"""
        print("init_slam_system: ", curr_shot.name)
        if (not self.system_initialized):
            rec_init, graph, matches = self.slam_init.initialize(curr_shot)
            self.system_initialized = (rec_init is not None)
            if self.system_initialized:
                self.slam_mapper.create_init_map(graph, rec_init,
                                                 self.slam_init.init_shot,
                                                 curr_shot)
                self.slam_mapper.velocity = np.eye(4)
            if self.system_initialized:
                print("Initialized with ", curr_shot.name)
                return True

    def track_frame(self, curr_shot: pymap.Shot):
        """ Tracks a frame
        """
        data = self.data
        logger.debug("Tracking: {}, {}".format(curr_shot.id, curr_shot.name))
        # Maybe move most of the slam_mapper stuff to tracking
        # TODO: Landmark replac!
        # Maybe not even necessary!
        # self.map.apply_landmark_replace(self.slam_mapper.last_shot)
        return self.slam_tracker.track(self.slam_mapper, curr_shot,
                                       self.config, self.camera, data)

        return self.system_initialized
