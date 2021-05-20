import numpy as np
import cv2
import pyopengv
import logging

from timeit import default_timer as timer
from collections import defaultdict

from opensfm import csfm
from opensfm import context
from opensfm import log
from opensfm import multiview
from opensfm import pairs_selection
from opensfm import feature_loader


logger = logging.getLogger(__name__)


def clear_cache():
    feature_loader.instance.clear_cache()


def match_images(data, ref_images, cand_images, overwrite):
    """ Perform pair matchings between two sets of images.

    It will do matching for each pair (i, j), i being in
    ref_images and j in cand_images, taking assumption that
    matching(i, j) == matching(j ,i). This does not hold for
    non-symmetric matching options like WORDS. Data will be
    stored in i matching only.

    If 'overwrite' is set to True, matches of a given images will be
    overwritten with the new ones, if False, they're going to be updated,
    keeping the previous ones.
    """

    # Get EXIFs data
    all_images = list(set(ref_images+cand_images))
    exifs = {im: data.load_exif(im) for im in all_images}

    # Generate pairs for matching
    pairs, preport = pairs_selection.match_candidates_from_metadata(
        ref_images, cand_images, exifs, data)
    logger.info('Matching {} image pairs'.format(len(pairs)))

    # Store per each image in ref for processing
    per_image = {im: [] for im in ref_images}
    for im1, im2 in pairs:
        per_image[im1].append(im2)

    ctx = Context()
    ctx.data = data
    ctx.cameras = ctx.data.load_camera_models()
    ctx.exifs = exifs
    ctx.overwrite = overwrite
    args = list(match_arguments(per_image, ctx))

    # Perform all pair matchings in parallel
    start = timer()
    mem_per_process = 512
    jobs_per_process = 2
    processes = context.processes_that_fit_in_memory(data.config['processes'], mem_per_process)
    logger.info("Computing pair matching with %d processes" % processes)
    matches = context.parallel_map(match_unwrap_args, args, processes, jobs_per_process)
    logger.debug('Matched {} pairs in {} seconds.'.format(
        len(pairs), timer()-start))

    # Index results per pair
    pairs = {}
    for im1, im1_matches in matches:
        for im2, m in im1_matches.items():
            pairs[im1, im2] = m

    return pairs, preport


class Context:
    pass


def match_arguments(pairs, ctx):
    """ Generate arguments for parralel processing of pair matching """
    pairs = sorted(pairs.items(), key=lambda x: -len(x[1]))
    for im, candidates in pairs:
        yield im, candidates, ctx


def match_unwrap_args(args):
    """Wrapper for parallel processing of pair matching.

    Compute all pair matchings of a given image and save them.
    """
    log.setup()
    im1, candidates, ctx = args

    im1_matches = {}
    p1, f1, _ = feature_loader.instance.load_points_features_colors(ctx.data, im1)
    camera1 = ctx.cameras[ctx.exifs[im1]['camera']]

    for im2 in candidates:
        p2, f2, _ = feature_loader.instance.load_points_features_colors(ctx.data, im2)
        camera2 = ctx.cameras[ctx.exifs[im2]['camera']]

        im1_matches[im2] = match(im1, im2, camera1, camera2, ctx.data)

    num_matches = sum(1 for m in im1_matches.values() if len(m) > 0)
    logger.debug('Image {} matches: {} out of {}'.format(
        im1, num_matches, len(candidates)))

    all_im1_matches = {} if ctx.overwrite else ctx.data.load_matches(im1)
    all_im1_matches.update(im1_matches)
    ctx.data.save_matches(im1, all_im1_matches)
    return im1, im1_matches


def match(im1, im2, camera1, camera2, data):
    """Perform matching for a pair of images."""
    # Apply mask to features if any
    time_start = timer()
    p1, f1, _ = feature_loader.instance.load_points_features_colors(
        data, im1, masked=True)
    p2, f2, _ = feature_loader.instance.load_points_features_colors(
        data, im2, masked=True)

    if p1 is None or p2 is None:
        return []

    config = data.config
    matcher_type = config['matcher_type'].upper()

    w1, w2 = None, None
    if 'WORDS' in matcher_type:
        w1 = feature_loader.instance.load_words(data, im1, masked=True)
        w2 = feature_loader.instance.load_words(data, im2, masked=True)
        if w1 is None or w2 is None:
            return []

    if matcher_type == 'WORDS':
        matches = csfm.match_using_words(
            f1, w1, f2, w2[:, 0],
            data.config['lowes_ratio'],
            data.config['bow_num_checks'])
    elif matcher_type == 'WORDS_SYMMETRIC':
        matches = match_words_symmetric(f1, w1, f2, w2, config)
    elif 'FLANN' in matcher_type:
        i1 = feature_loader.instance.load_features_index(data, im1, masked=True)
        i2 = feature_loader.instance.load_features_index(data, im2, masked=True)
        matches = match_flann_symmetric(f1, i1, f2, i2, config)
    elif matcher_type == 'BRUTEFORCE':
        matches = match_brute_force_symmetric(f1, f2, config)
    else:
        raise ValueError("Invalid matcher_type: {}".format(matcher_type))

    # Adhoc filters
    if config['matching_use_filters']:
        matches = apply_adhoc_filters(data, matches,
                                      im1, camera1, p1,
                                      im2, camera2, p2)

    matches = np.array(matches, dtype=int)
    time_2d_matching = timer() - time_start
    t = timer()

    robust_matching_min_match = config['robust_matching_min_match']
    if len(matches) < robust_matching_min_match:
        logger.debug(
            'Matching {} and {}.  Matcher: {} T-desc: {:1.3f} '
            'Matches: FAILED'.format(im1, im2, matcher_type, time_2d_matching))
        return []

    # robust matching
    rmatches = robust_match(p1, p2, camera1, camera2, matches, config)
    rmatches = np.array([[a, b] for a, b in rmatches])
    time_robust_matching = timer() - t
    time_total = timer() - time_start

    # From indexes in filtered sets, to indexes in original sets of features
    m1 = feature_loader.instance.load_mask(data, im1)
    m2 = feature_loader.instance.load_mask(data, im2)
    if m1 is not None and m2 is not None:
        rmatches = unfilter_matches(rmatches, m1, m2)

    logger.debug(
        'Matching {} and {}.  Matcher: {} '
        'T-desc: {:1.3f} T-robust: {:1.3f} T-total: {:1.3f} '
        'Matches: {} Robust: {} Success: {}'.format(
            im1, im2, matcher_type,
            time_2d_matching, time_robust_matching, time_total,
            len(matches), len(rmatches),
            len(rmatches) >= robust_matching_min_match))

    if len(rmatches) < robust_matching_min_match:
        return []

    return np.array(rmatches, dtype=int)


def match_words(f1, words1, f2, words2, config):
    """Match using words and apply Lowe's ratio filter.

    Args:
        f1: feature descriptors of the first image
        w1: the nth closest words for each feature in the first image
        f2: feature descriptors of the second image
        w2: the nth closest words for each feature in the second image
        config: config parameters
    """
    ratio = config['lowes_ratio']
    num_checks = config['bow_num_checks']
    return csfm.match_using_words(f1, words1, f2, words2[:, 0],
                                  ratio, num_checks)


def match_words_symmetric(f1, words1, f2, words2, config):
    """Match using words in both directions and keep consistent matches.

    Args:
        f1: feature descriptors of the first image
        w1: the nth closest words for each feature in the first image
        f2: feature descriptors of the second image
        w2: the nth closest words for each feature in the second image
        config: config parameters
    """
    matches_ij = match_words(f1, words1, f2, words2, config)
    matches_ji = match_words(f2, words2, f1, words1, config)
    matches_ij = [(a, b) for a, b in matches_ij]
    matches_ji = [(b, a) for a, b in matches_ji]

    return list(set(matches_ij).intersection(set(matches_ji)))


def match_flann(index, f2, config):
    """Match using FLANN and apply Lowe's ratio filter.

    Args:
        index: flann index if the first image
        f2: feature descriptors of the second image
        config: config parameters
    """
    search_params = dict(checks=config['flann_checks'])
    results, dists = index.knnSearch(f2, 2, params=search_params)
    squared_ratio = config['lowes_ratio']**2  # Flann returns squared L2 distances
    good = dists[:, 0] < squared_ratio * dists[:, 1]
    return list(zip(results[good, 0], good.nonzero()[0]))


def match_flann_symmetric(fi, indexi, fj, indexj, config):
    """Match using FLANN in both directions and keep consistent matches.

    Args:
        fi: feature descriptors of the first image
        indexi: flann index if the first image
        fj: feature descriptors of the second image
        indexj: flann index of the second image
        config: config parameters
    """
    matches_ij = [(a, b) for a, b in match_flann(indexi, fj, config)]
    matches_ji = [(b, a) for a, b in match_flann(indexj, fi, config)]

    return list(set(matches_ij).intersection(set(matches_ji)))


def match_brute_force(f1, f2, config):
    """Brute force matching and Lowe's ratio filtering.

    Args:
        f1: feature descriptors of the first image
        f2: feature descriptors of the second image
        config: config parameters
    """
    assert(f1.dtype.type == f2.dtype.type)
    if (f1.dtype.type == np.uint8):
        matcher_type = 'BruteForce-Hamming'
    else:
        matcher_type = 'BruteForce'
    matcher = cv2.DescriptorMatcher_create(matcher_type)
    matches = matcher.knnMatch(f1, f2, k=2)

    ratio = config['lowes_ratio']
    good_matches = []
    for match in matches:
        if match and len(match) == 2:
            m, n = match
            if m.distance < ratio * n.distance:
                good_matches.append(m)
    return _convert_matches_to_vector(good_matches)


def _convert_matches_to_vector(matches):
    """Convert Dmatch object to matrix form."""
    matches_vector = np.zeros((len(matches), 2), dtype=np.int)
    k = 0
    for mm in matches:
        matches_vector[k, 0] = mm.queryIdx
        matches_vector[k, 1] = mm.trainIdx
        k = k + 1
    return matches_vector


def match_brute_force_symmetric(fi, fj, config):
    """Match with brute force in both directions and keep consistent matches.

    Args:
        fi: feature descriptors of the first image
        fj: feature descriptors of the second image
        config: config parameters
    """
    matches_ij = [(a, b) for a, b in match_brute_force(fi, fj, config)]
    matches_ji = [(b, a) for a, b in match_brute_force(fj, fi, config)]

    return list(set(matches_ij).intersection(set(matches_ji)))


def robust_match_fundamental(p1, p2, matches, config):
    """Filter matches by estimating the Fundamental matrix via RANSAC."""
    if len(matches) < 8:
        return None, np.array([])

    p1 = p1[matches[:, 0]][:, :2].copy()
    p2 = p2[matches[:, 1]][:, :2].copy()

    FM_RANSAC = cv2.FM_RANSAC if context.OPENCV3 else cv2.cv.CV_FM_RANSAC
    threshold = config['robust_matching_threshold']
    F, mask = cv2.findFundamentalMat(p1, p2, FM_RANSAC, threshold, 0.9999)
    inliers = mask.ravel().nonzero()

    if F is None or F[2, 2] == 0.0:
        return F, []

    return F, matches[inliers]


def _compute_inliers_bearings(b1, b2, T, threshold=0.01):
    R = T[:, :3]
    t = T[:, 3]
    p = pyopengv.triangulation_triangulate(b1, b2, t, R)

    br1 = p.copy()
    br1 /= np.linalg.norm(br1, axis=1)[:, np.newaxis]

    br2 = R.T.dot((p - t).T).T
    br2 /= np.linalg.norm(br2, axis=1)[:, np.newaxis]

    ok1 = multiview.vector_angle_many(br1, b1) < threshold
    ok2 = multiview.vector_angle_many(br2, b2) < threshold
    return ok1 * ok2


def robust_match_calibrated(p1, p2, camera1, camera2, matches, config):
    """Filter matches by estimating the Essential matrix via RANSAC."""

    if len(matches) < 8:
        return np.array([])

    p1 = p1[matches[:, 0]][:, :2].copy()
    p2 = p2[matches[:, 1]][:, :2].copy()
    b1 = camera1.pixel_bearing_many(p1)
    b2 = camera2.pixel_bearing_many(p2)

    threshold = config['robust_matching_calib_threshold']
    T = multiview.relative_pose_ransac(
        b1, b2, b"STEWENIUS", 1 - np.cos(threshold), 1000, 0.999)

    for relax in [4, 2, 1]:
        inliers = _compute_inliers_bearings(b1, b2, T, relax * threshold)
        if sum(inliers) < 8:
            return np.array([])
        T = pyopengv.relative_pose_optimize_nonlinear(
            b1[inliers], b2[inliers], T[:3, 3], T[:3, :3])

    inliers = _compute_inliers_bearings(b1, b2, T, threshold)

    return matches[inliers]


def robust_match(p1, p2, camera1, camera2, matches, config):
    """Filter matches by fitting a geometric model.

    If cameras are perspective without distortion, then the Fundamental
    matrix is used.  Otherwise, we use the Essential matrix.
    """
    if (camera1.projection_type == 'perspective'
            and camera1.k1 == 0.0 and camera1.k2 == 0.0
            and camera2.projection_type == 'perspective'
            and camera2.k1 == 0.0 and camera2.k2 == 0.0):
        return robust_match_fundamental(p1, p2, matches, config)[1]
    else:
        return robust_match_calibrated(p1, p2, camera1, camera2, matches, config)


def unfilter_matches(matches, m1, m2):
    """Given matches and masking arrays, get matches with un-masked indexes."""
    i1 = np.flatnonzero(m1)
    i2 = np.flatnonzero(m2)
    return np.array([(i1[match[0]], i2[match[1]]) for match in matches])


def apply_adhoc_filters(data, matches, im1, camera1, p1, im2, camera2, p2):
    """ Apply a set of filters functions defined further below
        for removing static data in images.

    """
    matches = _non_static_matches(p1, p2, matches, data.config)
    matches = _not_on_pano_poles_matches(p1, p2, matches, camera1, camera2)
    matches = _not_on_vermont_watermark(p1, p2, matches, im1, im2, data)
    matches = _not_on_blackvue_watermark(p1, p2, matches, im1, im2, data)
    return matches


def _non_static_matches(p1, p2, matches, config):
    """Remove matches with same position in both images.

    That should remove matches on that are likely belong to rig occluders,
    watermarks or dust, but not discard entirely static images.
    """
    threshold = 0.001
    res = []
    for match in matches:
        d = p1[match[0]] - p2[match[1]]
        if d[0]**2 + d[1]**2 >= threshold**2:
            res.append(match)

    static_ratio_threshold = 0.85
    static_ratio_removed = 1 - len(res) / max(len(matches), 1)
    if static_ratio_removed > static_ratio_threshold:
        return matches
    else:
        return res


def _not_on_pano_poles_matches(p1, p2, matches, camera1, camera2):
    """Remove matches for features that are too high or to low on a pano.

    That should remove matches on the sky and and carhood part of panoramas
    """
    min_lat = -0.125
    max_lat = 0.125
    is_pano1 = (camera1.projection_type == 'equirectangular')
    is_pano2 = (camera2.projection_type == 'equirectangular')
    if is_pano1 or is_pano2:
        res = []
        for match in matches:
            if ((not is_pano1 or min_lat < p1[match[0]][1] < max_lat) and
                    (not is_pano2 or min_lat < p2[match[1]][1] < max_lat)):
                res.append(match)
        return res
    else:
        return matches


def _not_on_vermont_watermark(p1, p2, matches, im1, im2, data):
    """Filter Vermont images watermark."""
    meta1 = data.load_exif(im1)
    meta2 = data.load_exif(im2)

    if meta1['make'] == 'VTrans_Camera' and meta1['model'] == 'VTrans_Camera':
        matches = [m for m in matches if _vermont_valid_mask(p1[m[0]])]
    if meta2['make'] == 'VTrans_Camera' and meta2['model'] == 'VTrans_Camera':
        matches = [m for m in matches if _vermont_valid_mask(p2[m[1]])]
    return matches


def _vermont_valid_mask(p):
    """Check if pixel inside the valid region.

    Pixel coord Y should be larger than 50.
    In normalized coordinates y > (50 - h / 2) / w
    """
    return p[1] > -0.255


def _not_on_blackvue_watermark(p1, p2, matches, im1, im2, data):
    """Filter Blackvue's watermark."""
    meta1 = data.load_exif(im1)
    meta2 = data.load_exif(im2)

    if meta1['make'].lower() == 'blackvue':
        matches = [m for m in matches if _blackvue_valid_mask(p1[m[0]])]
    if meta2['make'].lower() == 'blackvue':
        matches = [m for m in matches if _blackvue_valid_mask(p2[m[1]])]
    return matches


def _blackvue_valid_mask(p):
    """Check if pixel inside the valid region.

    Pixel coord Y should be smaller than h - 70.
    In normalized coordinates y < (h - 70 - h / 2) / w,
    with h = 2160 and w = 3840
    """
    return p[1] < 0.263
