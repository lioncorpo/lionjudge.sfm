import os.path, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from slam_initializer import SlamInitializer
from slam_mapper import SlamMapper
from slam_tracker import SlamTracker
from slam_types import Frame
from slam_types import Keyframe
import slam_debug
from opensfm import dataset
from opensfm import reconstruction
from opensfm import feature_loader
from opensfm import types
import numpy as np
from opensfm import feature_loading
import networkx as nx
from opensfm import log
import logging
import slam_config
log.setup()
logger = logging.getLogger(__name__)


class SlamSystem(object):

    def __init__(self, args):
        print("Init slam system", args)
        self.data = dataset.DataSet(args.dataset)
        cameras = self.data.load_camera_models()
        self.config = self.data.config
        self.camera = next(iter(cameras.items()))
        self.system_initialized = False
        self.system_lost = True
        self.reconstruction_init = None
        self.image_list = sorted(self.data.image_list)
        self.config_slam = slam_config.default_config()
        self.slam_mapper = SlamMapper(self.data, self.config,
                                      self.config_slam, self.camera)
        self.slam_tracker = SlamTracker(self.data, self.config)
        self.initializer = SlamInitializer(self.config)

    def add_arguments(self, parser):
        parser.add_argument('dataset', help='dataset to process')

    def init_slam_system(self, data, frame: Frame):
        """Find the initial depth estimates for the slam map"""
        print("init_slam_system: ", frame)
        if self.initializer.init_frame is None:
            self.initializer.set_initial_frame(data, frame)
            self.system_initialized = False
        else:
            rec_init, graph, matches = \
                self.initializer.initialize(data, self.config_slam, frame)
            self.system_initialized = (rec_init is not None)
            if self.system_initialized:
                self.slam_mapper.create_init_map(graph, rec_init,
                                                 self.initializer.init_frame,
                                                 frame,
                                                 self.initializer.init_pdc,
                                                 self.initializer.other_pdc)
        return self.system_initialized

    def track_next_frame_lk(self, data, frame: Frame):
        """Estimates the pose for the next frame"""
        chrono = reconstruction.Chronometer()
        if not self.system_initialized:
            frame.extract_features(data, self.config_slam['extract_features'])
            self.system_initialized = self.init_slam_system(data, frame)
            if self.system_initialized:
                chrono.lap('init')
                slam_debug.avg_timings.addTimes(chrono.laps_dict)
                logger.debug("Initialized system with {}".format(frame.im_name))
            else:
                logger.debug("Failed to initialize with {}".format(frame.im_name))
            return self.system_initialized
        else:
            pose = self.slam_tracker.track_LK(self.slam_mapper, frame,
                                    self.config, self.camera, data)
            chrono.lap('track_lk')
            slam_debug.avg_timings.addTimes(chrono.laps_dict)
            if pose is None:
                return False
            frame.world_pose = pose
            self.slam_mapper.update_with_last_frame(frame)
            if self.slam_mapper.new_kf_needed(frame):  # and False:
                self.slam_mapper.create_new_keyframe(frame)
                if self.slam_mapper.n_keyframes % 5 == 0:
                    self.slam_mapper.paint_reconstruction(data)
                    self.slam_mapper.\
                        save_reconstruction(data, frame.im_name+"aft")
                logger.debug("Inserting new KF: {}".format(frame.im_name))
            else:
                print("No kf needed")
            return True
        return False

    def track_next_frame(self, data, frame: Frame):
        """Estimates the pose for the next frame"""
        chrono = reconstruction.Chronometer()
        if not self.system_initialized:
            self.system_initialized = self.init_slam_system(data, frame)
            if self.system_initialized:
                chrono.lap('init')
                slam_debug.avg_timings.addTimes(chrono.laps_dict)
                logger.debug("Initialized system with {}".format(frame.im_name))
            else:
                logger.debug("Failed to initialize with {}".format(frame.im_name))
            return self.system_initialized
        else:
            logger.debug("Tracking: {}, {}".format(frame.frame_id, frame.im_name))
            # Maybe move most of the slam_mapper stuff to tracking
            self.slam_mapper.apply_landmark_replace()
            pose = self.slam_tracker.track(self.slam_mapper, frame,
                                           self.config, self.camera, data)
            chrono.lap('track')
            slam_debug.avg_timings.addTimes(chrono.laps_dict)
            print("pose after track for ", frame.im_name, ": ",
                  pose.rotation, pose.translation)
            if pose is not None:
                frame.world_pose = pose
                self.slam_mapper.update_local_map(frame)
                print("After update_local_map")
                chrono.start()
                if self.config_slam["refine_with_local_map"]:
                    pose: types.Pose = self.slam_mapper.\
                        track_with_local_map(frame, self.slam_tracker)
                    chrono.lap('track_local_map')
                if pose is not None:
                    self.slam_mapper.set_last_frame(frame)
                    if self.slam_mapper.new_kf_needed(frame):
                        new_kf = Keyframe(frame, data,
                                          self.slam_mapper.n_keyframes, frame.load_points_desc_colors())
                        self.slam_mapper.add_keyframe(new_kf)
                        self.slam_mapper.mapping_with_new_keyframe(new_kf)
                        self.slam_mapper.local_bundle_adjustment()
                        print("local ba") 
                        if new_kf.kf_id % 5 == 0:
                            self.slam_mapper.paint_reconstruction(data)
                            self.slam_mapper.\
                                save_reconstruction(data, frame.im_name+"aft")
                        logger.debug("Inserting new KF: {}".format(new_kf.im_name))
                    else:
                        print("No kf needed")
            slam_debug.avg_timings.addTimes(chrono.laps_dict)
            return True

    def reset(self):
        """Reset the system"""
        return True
