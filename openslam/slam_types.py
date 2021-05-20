from opensfm import types
from opensfm import feature_loader
from opensfm import io
import slam_utils
import cslam 
import numpy as np
class Landmark(object):

    def __init__(self, graph_id):
        """Creates a landmark

        Arguments:
            graph_id : The id the landmark has in the graph (must be unique)
        """
        self.lm_id = graph_id
        # self.is_observable_in_tracking = False
        # self.local_map_update_identifier = -1  # the last frame where it was observed
        # self.identifier_in_local_lm_search_ = -1
        # self.n_observable = 0  # the number of frames and KFs it is seen in
        # self.descriptor = None
        # self.num_observable = 1  # landmarks visible in current frame (maybe without match)
        # self.num_observed = 1  # landmarks matched in current frame
        # self.first_kf_id = -1
        # self.ref_kf = None
        # self.mean_normal = []
        self.clm = None

    # def update_descriptor(self, desc):
    #     """ There are two options:
    #     - Update with most recent descriptor
    #     - or compute similar to openvslam
    #     """


class Frame(object):
    """Every image form the input is a Frame.
    Only selected frames become keyframes
    """
    def __init__(self, name, id, data):
        self.im_name = name
        self.image = io.imread(data._image_file(name), grayscale=True)  # The gray-scale image
        self.frame_id = id
        self.kf_id = -1  # if non-KF, id of "parent" KF
        self.world_pose = types.Pose()
        self.rel_pose_to_kf = types.Pose()

        self.cframe = None

        self.img_pyr = [self.image]

        self.has_features = False
        self.descriptors = None
        self.points = None
        self.colors = None

        self.undist_pts = None
        self.keypts_in_cell = None

    def make_cframe(self, orb_extractor):
        mask = np.array([], dtype=np.uint8)
        self.cframe =\
            cslam.Frame(self.image, mask, self.im_name, self.frame_id, orb_extractor)

    def load_points_desc_colors(self):
        if self.has_features:
            return (self.points, self.descriptors, self.colors)
        return None

    def prepare_for_storage(self):
        """Called before storing frame in the mapper.
        Cleans up memory by releasing image, desc, points, colors
        """
        self.has_features = False
        self.descriptors, self.points, self.colors = None, None, None
        self.image = None

    def extract_features(self, data, do_extract=False):
        """Loads or extracts descriptors, points and colors"""
        if do_extract:
            self.points, self.descriptors, self.colors =\
                slam_utils.extract_features(self.im_name, data)
        else:
            self.points, self.descriptors, self.colors = \
                feature_loader.instance.load_points_features_colors(
                    data, self.im_name, masked=True)

        self.has_features = self.points is not None\
            and self.descriptors is not None\
            and self.colors is not None


class Keyframe(object):
    def __init__(self, frame: Frame, data, kf_id):
        """Initialize a new keyframe with a frame and a kf_id
        The Keyframe holds a frame!
        """
        # print("Creating KF: ", kf_id, frame.im_name)
        self.kf_id = kf_id  # unique_id
        # self.frame = frame
        self.frame_id = frame.frame_id
        self.colors = frame.colors
        self.im_name = frame.im_name
        # self.frame.image = None  # free image
        self.world_pose = frame.world_pose
        self.rel_pose_to_kf = types.Pose()  # since this is a keyframe, rel = I
        self.ckf = None
