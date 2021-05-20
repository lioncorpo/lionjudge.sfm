from opensfm import matching
# from feature_loader import instance
from opensfm import feature_loader
from opensfm import features
import logging
import numpy as np
from slam_types import Frame
from slam_types import Keyframe
import slam_debug
from opensfm import reconstruction
logger = logging.getLogger(__name__)

"""The SlamMatcher matches a keyframe to the current frame and return the matches
"""

def match(data, ref_frame: str, curr_frame: str, camera):
    print("Matching!", ref_frame, curr_frame)
    im1_matches = matching.match(ref_frame, curr_frame,
                                    camera, camera, data)
    if len(im1_matches) < 30:
        return False, []
    return True, im1_matches

def match_desc_and_points(data, f1, f2, p1, p2, camera):
    if f1 is None or f2 is None:
        return []
    config = data.config
    matcher_type = config['matcher_type'].upper()
    symmetric_matching = config['symmetric_matching']   
    if matcher_type == 'BRUTEFORCE':
        if symmetric_matching:
            matches = matching.match_brute_force_symmetric(f1, f2, config)
        else:
            matches = matching.match_brute_force(f1, f2, config)
    if len(matches) < 30:
        return False, []
    matches = np.array(matches, dtype=int)
    symmetric = 'symmetric' if config['symmetric_matching'] \
        else 'one-way'
    robust_matching_min_match = config['robust_matching_min_match']
    if len(matches) < robust_matching_min_match:
        logger.debug(
            'Matching {} and {}.  Matcher: {} ({}) T-desc: {:1.3f} '
            'Matches: FAILED'.format(
                im1, im2,
                matcher_type, symmetric))
        return False, []

    # robust matching
    rmatches = matching.robust_match(p1, p2, camera, camera, matches, config)
    rmatches = np.array([[a, b] for a, b in rmatches])
    if len(rmatches) < 30:
        return False, []
    return True, rmatches


def match_desc_desc(f1, f2, data):
    if f1 is None or f2 is None:
        return np.array([])
    if len(f1) == 0 or len(f2) == 0:
        return np.array([])
    chrono = reconstruction.Chronometer()
    matches = matching.match_brute_force_symmetric(f1, f2, data.config)
    # matches_bf = matching.match_brute_force(f1,f2, data.config)
    # print("matches: ", len(matches), " matches_bf", len(matches_bf))
    chrono.lap('frame_to_lm')
    slam_debug.avg_timings.addTimes(chrono.laps_dict)
    return np.asarray(matches, dtype=int)  # len(matches), matches


def match_frame_to_landmarks(f1, landmarks,
                             margin, data, graph):
    """Matches features f1 to landmarks"""
    f2 = []
    for lm_id in landmarks:
        lm = graph.node[lm_id]['data']
        if lm.descriptor is not None:
            f2.append(lm.descriptor)
    f2 = np.asarray(f2)
    return match_desc_desc(f1, f2, data)


def match_for_triangulation(curr_kf: Keyframe,
                            other_kf: Keyframe, graph, data):
    cameras = data.load_camera_models()
    camera_obj = next(iter(cameras.values()))
    print("match_for_triangulation", other_kf, curr_kf.im_name)
    p1, f1, _ = curr_kf.load_points_desc_colors()
    p2, f2, _ = other_kf.load_points_desc_colors()
    success, matches = match_desc_and_points(data, f1, f2, p1, p2, camera_obj)

    return matches if success else None

    # def match_for_triangulation_fast(self, curr_kf: Keyframe,
    #                                  other_kf: Keyframe, graph, data):
    #     cameras = data.load_camera_models()
    #     camera_obj = next(iter(cameras.values()))
    #     print("match_for_triangulation", other_kf.im_name, curr_kf.im_name)
    #     f1, f2 = curr_kf.descriptors, other_kf.descriptors
    #     i1, i2 = curr_kf.index, other_kf.index
    #     p1, p2 = curr_kf.points, other_kf.points
    #     config = data.config
    #     matches = matching.match_flann_symmetric(f1, i1, f2, i2, config)
    #     if matches is None:
    #         return None
    #     matches = np.asarray(matches)
    #     rmatches = matching.robust_match(p1, p2, camera_obj, camera_obj, 
    #                                      matches, config)
    #     rmatches = np.array([[a, b] for a, b in rmatches])
    #     print("n_matches {} <-> {}: {}, {}".format(
    #           curr_kf.im_name, other_kf,
    #           len(matches), len(rmatches)))
    #     # From indexes in filtered sets, to indexes in original sets of features
    #     m1 = feature_loader.instance.load_mask(data, curr_kf.im_name)
    #     m2 = feature_loader.instance.load_mask(data, other_kf.im_name)
    #     if m1 is not None and m2 is not None:
    #         rmatches = matching.unfilter_matches(rmatches, m1, m2)
    #     return np.array(rmatches, dtype=int)

# def matchOpenVSlam(self):
#     #think about the matching.
#     #reproject landmarks visible in last frame to current frame
#     #under a velocity model
#     #openvslam, projection.cc, l84
#     #reproject to image -> check inside
#     #find features in cell
#     #hamming matching
#     return True

