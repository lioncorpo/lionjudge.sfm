import logging
from itertools import combinations
from collections import defaultdict
import math
import random
import numpy as np
import cv2
import os.path

import scipy.spatial as spatial

from opensfm import bow
from opensfm import context
from opensfm import feature_loader

logger = logging.getLogger(__name__)


def has_gps_info(exif):
    return (exif and
            'gps' in exif and
            'latitude' in exif['gps'] and
            'longitude' in exif['gps'])


def match_candidates_by_distance(images_ref, images_cand, exifs, reference,
                                 max_neighbors, max_distance):
    """Find candidate matching pairs by GPS distance.

    The GPS altitude is ignored because we want images of the same position
    at different altitudes to be matched together.  Otherwise, for drone
    datasets, flights at different altitudes do not get matched.
    """
    if len(images_cand) == 0:
        return set()
    if max_neighbors <= 0 and max_distance <= 0:
        return set()
    max_neighbors = max_neighbors or 99999999
    max_distance = max_distance or 99999999.
    k = min(len(images_cand), max_neighbors)

    points = np.zeros((len(images_cand), 3))
    for i, image in enumerate(images_cand):
        gps = exifs[image]['gps']
        points[i] = reference.to_topocentric(
            gps['latitude'], gps['longitude'], 0)

    tree = spatial.cKDTree(points)

    pairs = set()
    for image_ref in images_ref:
        nn = k+1 if image_ref in images_cand else k

        gps = exifs[image_ref]['gps']
        point = reference.to_topocentric(
            gps['latitude'], gps['longitude'], 0)
        distances, neighbors = tree.query(
            point, k=nn, distance_upper_bound=max_distance)

        if type(neighbors) == int:      # special case with only one NN
            neighbors = [neighbors]

        for j in neighbors:
            if j >= len(images_cand):
                continue
            image_cand = images_cand[j]
            if image_cand != image_ref:
                pairs.add(tuple(sorted((image_ref, image_cand))))
    return pairs


def match_candidates_with_bow(data, images_ref, images_cand,
                              exifs, reference, max_neighbors,
                              max_gps_distance, max_gps_neighbors,
                              enforce_other_cameras):
    """Find candidate matching pairs using BoW-based distance.

    If max_gps_distance > 0, then we use first restrain a set of
    candidates using max_gps_neighbors neighbors selected using
    GPS distance.

    If enforce_other_cameras is True, we keep max_neighbors images
    with same cameras AND max_neighbors images from any other different
    camera.
    """
    if max_neighbors <= 0:
        return set()

    preempted_candidates, need_load = preempt_candidates(
            images_ref, images_cand,
            exifs, reference,
            max_gps_neighbors, max_gps_distance)

    # construct BoW histograms
    logger.info("Computing %d BoW histograms" % len(need_load))
    histograms = load_histograms(data, need_load)

    # parallel VLAD neighbors computation
    args, processes, batch_size = create_parallel_matching_args(
        data, preempted_candidates, histograms)
    logger.info("Computing BoW candidates with %d processes" % processes)
    results = context.parallel_map(match_bow_unwrap_args, args, processes, batch_size)

    return construct_pairs(results, max_neighbors, exifs, enforce_other_cameras)


def match_candidates_with_vlad(data, images_ref, images_cand,
                               exifs, reference, max_neighbors,
                               max_gps_distance, max_gps_neighbors,
                               enforce_other_cameras):
    """Find candidate matching pairs using VLAD-based distance.
     If max_gps_distance > 0, then we use first restrain a set of
    candidates using max_gps_neighbors neighbors selected using
    GPS distance.

    If enforce_other_cameras is True, we keep max_neighbors images
    with same cameras AND max_neighbors images from any other different
    camera.
    """
    if max_neighbors <= 0:
        return set()

    preempted_candidates, need_load = preempt_candidates(
            images_ref, images_cand,
            exifs, reference,
            max_gps_neighbors, max_gps_distance)

    # construct VLAD histograms
    logger.info("Computing %d VLAD histograms" % len(need_load))
    histograms = vlad_histograms(need_load, data)

    # parallel VLAD neighbors computation
    args, processes, batch_size = create_parallel_matching_args(
        data, preempted_candidates, histograms)
    logger.info("Computing VLAD candidates with %d processes" % processes)
    results = context.parallel_map(match_vlad_unwrap_args, args, processes, batch_size)

    return construct_pairs(results, max_neighbors, exifs, enforce_other_cameras)


def preempt_candidates(images_ref, images_cand,
                       exifs, reference,
                       max_gps_neighbors, max_gps_distance):
    """Preempt candidates using GPS to reduce set of images
    from which to load data to save RAM.
    """

    # preempt candidates images using GPS
    preempted_cand = {im: images_cand for im in images_ref}
    if max_gps_distance > 0 or max_gps_neighbors > 0:
        gps_pairs = match_candidates_by_distance(images_ref, images_cand,
                                                 exifs, reference,
                                                 max_gps_neighbors,
                                                 max_gps_distance)
        preempted_cand = defaultdict(list)
        for p in gps_pairs:
            if p[0] in images_ref:
                preempted_cand[p[0]].append(p[1])
            if p[1] in images_ref:
                preempted_cand[p[1]].append(p[0])

    # reduce sets of images from which to load histograms (RAM saver)
    need_load = set(preempted_cand.keys())
    for k, v in preempted_cand.items():
        need_load.update(v)
        need_load.add(k)
    return preempted_cand, need_load


def construct_pairs(results, max_neighbors, exifs, enforce_other_cameras):
    """Construct final sets of pairs to match"""
    pairs = set()
    for im, order, other in results:
        if enforce_other_cameras:
            pairs = pairs.union(pairs_from_neighbors(im, exifs, order, other, max_neighbors))
        else:
            for i in order[:max_neighbors]:
                pairs.add(tuple(sorted((im, other[i]))))
    return pairs


def vlad_histograms(images, data, vlad_count):
    if len(images) == 0:
        return {}

    desc_size = 128
    vlads = vlad_vocabulary()
    vlad_count = 64

    image_vlads = {}
    for im in images:
        _, features, _ = data.load_features(im)
        m = feature_loader.load_masks(data, im)
        features = features if m is None else features[m]

        vlad = unnormalized_vlad(features, vlads, vlad_count, desc_size)
        vlad = signed_square_root_normalize(vlad)
        image_vlads[im] = vlad

    return image_vlads


def random_single_image_vocabulary(data, vlad_count):
    # Random feature from a single image
    _, vlads, _ = data.load_features(data.images()[0])
    np.random.shuffle(vlads)
    vlads = vlads[:vlad_count]
    return vlads


def random_image_vocabulary(data, vlad_count):
    # Random features from all images
    c = int(math.ceil(float(vlad_count) / len(images)))
    vlads = []
    for im in images:
        _, f, _ = data.load_features(im)
        indices = random.sample(range(0, len(f)), c)
        vlads.extend(f[indices])

    np.random.shuffle(vlads)
    vlads = np.array(vlads)[:vlad_count]
    return vlads


def random_bow_vocabulary(data, vlad_count):
    # Random words from the bow vocabulary
    vlads, _ = bow.load_bow_words_and_frequencies(data.config)
    np.random.shuffle(vlads)
    vlads = vlads[:vlad_count]
    return vlads


def one_step_kmeans_vocabulary(data):
    samples = []
    for im in images:
        _, fs, _ = data.load_features(im)
        samples.extend(fs)

    samples = np.array(samples)

    max_iter = 1
    attempts = 1
    nclusters = vlad_count
    criteria = (cv2.TERM_CRITERIA_MAX_ITER, max_iter, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, _, vlads = cv2.kmeans(samples, nclusters, None, criteria, attempts, flags)
    return vlads


def vlad_vocabulary():
    vlads_file = os.path.join(context.BOW_PATH, 'bow_hahog_root_uchar_64.npz')
    vlads = np.load(vlads_file)['words']
    return vlads


def unnormalized_vlad(features, centers, vlad_count, desc_size):
    # Unnormalized VLAD
    vlad = np.zeros((vlad_count, desc_size), dtype=np.float32)
    for f in features:
        i = np.argmin(np.linalg.norm(f-centers, axis=1))
        vlad[i, :] += f-centers[i]
    vlad = np.ndarray.flatten(vlad)
    return vlad


def adapt_cluster_centers(features, centers, vlad_count):
    assignments = { i: [] for i in range(vlad_count) }
    for f in features:
        i = np.argmin(np.linalg.norm(f-centers, axis=1))
        assignments[i].append(f)

    adapted_centers = []
    for i in assignments:
        if not len(assignments[i]):
            continue

        adapted_centers.append(np.mean(np.array(assignments[i]), axis=0))

    return np.array(adapted_centers)

def intra_normalize(v, count, size):
    for i in range(0, count):
        n = np.linalg.norm(v[size * i:size * (i+1)])
        n = n if n > 0 else 1
        v[size * i:size * (i+1)] /= n

    return normalize(v)


def signed_square_root_normalize(v):
    # Signed square root (SSR) normalization
    v = np.sign(v) * np.sqrt(np.abs(v))
    return normalize(v)


def normalize(v):
    v /= np.linalg.norm(v)
    return v


def vlad_distances(image, other_images, histograms):
    """ Compute VLAD-based distance (L2 on VLAD-histogram)
        between an image and other images.
    """
    if image not in histograms:
        return image, [], []

    distances = []
    other = []
    h = histograms[image]
    for im2 in other_images:
        if im2 != image and im2 in histograms:
            h2 = histograms[im2]
            distances.append(np.linalg.norm(h - h2))
            other.append(im2)
    return image, np.argsort(distances), other


def create_parallel_matching_args(data, preempted_cand, histograms):
    """Create arguments to matching function"""
    args = list(match_histogram_arguments(preempted_cand, histograms))

     # parallel VLAD neighbors computation
    per_process = 512
    processes = context.processes_that_fit_in_memory(data.config['processes'], per_process)
    batch_size = max(1, len(args)/(2*processes))
    return args, processes, batch_size


def match_histogram_arguments(candidates, histograms):
    """ Generate arguments for parralel processing of BoW """
    for im, cands in candidates.items():
        yield (im, cands, histograms)


def match_bow_unwrap_args(args):
    """ Wrapper for parralel processing of BoW """
    image, other_images, histograms = args
    return bow_distances(image, other_images, histograms)


def match_vlad_unwrap_args(args):
    """ Wrapper for parralel processing of VLAD """
    image, other_images, histograms = args
    return vlad_distances(image, other_images, histograms)


def match_candidates_by_time(images_ref, images_cand, exifs, max_neighbors):
    """Find candidate matching pairs by time difference."""
    if max_neighbors <= 0:
        return set()
    k = min(len(images_cand), max_neighbors)

    times = np.zeros((len(images_cand), 1))
    for i, image in enumerate(images_cand):
        times[i] = exifs[image]['capture_time']

    tree = spatial.cKDTree(times)

    pairs = set()
    for image_ref in images_ref:
        nn = k+1 if image_ref in images_cand else k

        time = exifs[image_ref]['capture_time']
        distances, neighbors = tree.query([time], k=nn)

        if type(neighbors) == int:      # special case with only one NN
            neighbors = [neighbors]

        for j in neighbors:
            if j >= len(images_cand):
                continue
            image_cand = images_cand[j]
            if image_ref != image_cand:
                pairs.add(tuple(sorted((image_ref, image_cand))))
    return pairs


def match_candidates_by_order(images_ref, images_cand, max_neighbors):
    """Find candidate matching pairs by sequence order."""
    if max_neighbors <= 0:
        return set()
    n = (max_neighbors + 1) // 2

    pairs = set()
    for i, image_ref in enumerate(images_ref):
        a = max(0, i - n)
        b = min(len(images_cand), i + n)
        for j in range(a, b):
            image_cand = images_cand[j]
            if image_ref != image_cand:
                pairs.add(tuple(sorted([image_ref, image_cand])))
    return pairs


def match_candidates_from_metadata(images_ref, images_cand, exifs, data):
    """Compute candidate matching pairs between between images_ref and images_cand

    Returns a list of pairs (im1, im2) such that (im1 in images_ref) is true.
    Returned pairs are unique given that (i, j) == (j, i).
    """
    max_distance = data.config['matching_gps_distance']
    gps_neighbors = data.config['matching_gps_neighbors']
    time_neighbors = data.config['matching_time_neighbors']
    order_neighbors = data.config['matching_order_neighbors']
    bow_neighbors = data.config['matching_bow_neighbors']
    bow_gps_distance = data.config['matching_bow_gps_distance']
    bow_gps_neighbors = data.config['matching_bow_gps_neighbors']
    bow_other_cameras = data.config['matching_bow_other_cameras']
    vlad_neighbors = data.config['matching_vlad_neighbors']
    vlad_gps_distance = data.config['matching_vlad_gps_distance']
    vlad_gps_neighbors = data.config['matching_vlad_gps_neighbors']
    vlad_other_cameras = data.config['matching_vlad_other_cameras']

    if not data.reference_lla_exists():
        data.invent_reference_lla()
    reference = data.load_reference()

    if not all(map(has_gps_info, exifs.values())):
        if gps_neighbors != 0:
            logger.warn("Not all images have GPS info. "
                        "Disabling matching_gps_neighbors.")
        gps_neighbors = 0
        max_distance = 0

    images_ref.sort()

    if max_distance == gps_neighbors == time_neighbors == order_neighbors == bow_neighbors == vlad_neighbors == 0:
        # All pair selection strategies deactivated so we match all pairs
        d = set()
        t = set()
        o = set()
        b = set()
        v = set()
        pairs = set([tuple(sorted([i, j])) for i in images_ref for j in images_cand])
    else:
        d = match_candidates_by_distance(images_ref, images_cand, exifs, reference,
                                         gps_neighbors, max_distance)
        t = match_candidates_by_time(images_ref, images_cand, exifs, time_neighbors)
        o = match_candidates_by_order(images_ref, images_cand, order_neighbors)
        b = match_candidates_with_bow(data, images_ref, images_cand,
                                      exifs, reference, bow_neighbors,
                                      bow_gps_distance, bow_gps_neighbors,
                                      bow_other_cameras)
        v = match_candidates_with_vlad(data, images_ref, images_cand,
                                       exifs, reference, vlad_neighbors,
                                       vlad_gps_distance, vlad_gps_neighbors,
                                       vlad_other_cameras)
        pairs = d | t | o | b | v

    pairs = ordered_pairs(pairs, images_ref)

    report = {
        "num_pairs_distance": len(d),
        "num_pairs_time": len(t),
        "num_pairs_order": len(o),
        "num_pairs_bow": len(b),
        "num_pairs_vlad": len(v),
    }
    return pairs, report


def bow_distances(image, other_images, histograms):
    """ Compute BoW-based distance (L1 on histogram of words)
        between an image and other images.
    """
    if image not in histograms:
        return image, [], []

    distances = []
    other = []
    h = histograms[image]
    for im2 in other_images:
        if im2 != image and im2 in histograms:
            h2 = histograms[im2]
            distances.append(np.fabs(h - h2).sum())
            other.append(im2)
    return image, np.argsort(distances), other


def load_histograms(data, images):
    """ Load BoW histograms of given images """
    min_num_feature = 8

    histograms = {}
    bows = bow.load_bows(data.config)
    for im in images:
        filtered_words = feature_loader.instance.load_words(data, im, masked=True)
        if filtered_words is None:
            logger.error("No words in image {}".format(im))
            continue
        if len(filtered_words) <= min_num_feature:
            logger.warning("Too few filtered features in image {}: {}".format(
                im, len(filtered_words)))
            continue

        histograms[im] = bows.histogram(filtered_words[:, 0])
    return histograms


def vlad_histograms(images, data):
    """ Construct VLAD histograms from the image features.

        Returns a dictionary of VLAD vectors for the images.
    """
    if len(images) == 0:
        return {}

    words, _ = bow.load_vlad_words_and_frequencies(data.config)
    vlads = {}
    for im in images:
        _, features, _ = feature_loader.instance.load_points_features_colors(
            data, im, masked=True)
        vlad = unnormalized_vlad(features, words)
        vlad = signed_square_root_normalize(vlad)
        vlads[im] = vlad

    return vlads


def unnormalized_vlad(features, centers):
    """ Compute unnormalized VLAD histograms from a set of
        features in relation to centers.

        Returns the unnormalized VLAD vector.
    """
    vlad = np.zeros(centers.shape, dtype=np.float32)
    for f in features:
        i = np.argmin(np.linalg.norm(f-centers, axis=1))
        vlad[i, :] += f-centers[i]
    vlad = np.ndarray.flatten(vlad)
    return vlad


def signed_square_root_normalize(v):
    """ Compute Signed Square Root (SSR) normalization on
        a vector.

        Returns the SSR normalized vector.
    """
    v = np.sign(v) * np.sqrt(np.abs(v))
    v /= np.linalg.norm(v)
    return v


def vlad_distances(image, other_images, histograms):
    """ Compute VLAD-based distance (L2 on VLAD-histogram)
        between an image and other images.

        Returns the image, the order of the other images,
        and the other images.
    """
    if image not in histograms:
        return image, [], []

    distances = []
    other = []
    h = histograms[image]
    for im2 in other_images:
        if im2 != image and im2 in histograms:
            h2 = histograms[im2]
            distances.append(np.linalg.norm(h - h2))
            other.append(im2)
    return image, np.argsort(distances), other


def pairs_from_neighbors(image, exifs, order, other, max_neighbors):
    """Construct matching pairs given closest ordered neighbors.

    Pairs will of form (image, im2), im2 being the closest max_neighbors
    given by (order, other) having the same cameras OR the closest max_neighbors
    having from any other camera.
    """
    same_camera, other_cameras = [], []
    for i in order:
        im2 = other[i]
        if exifs[im2]['camera'] == exifs[image]['camera']:
            if len(same_camera) < max_neighbors:
                same_camera.append(im2)
        else:
            if len(other_cameras) < max_neighbors:
                other_cameras.append(im2)
        if len(same_camera) + len(other_cameras) >= 2 * max_neighbors:
            break

    pairs = set()
    for im2 in same_camera+other_cameras:
        pairs.add(tuple(sorted((image, im2))))
    return pairs


def ordered_pairs(pairs, images_ref):
    """Image pairs that need matching skipping duplicates.

    Returns a list of pairs (im1, im2) such that (im1 in images_ref) is true.
    """
    per_image = defaultdict(list)
    for im1, im2 in pairs:
        per_image[im1].append(im2)
        per_image[im2].append(im1)

    ordered = set()
    remaining = set(images_ref)
    if len(remaining) > 0:
        next_image = remaining.pop()
        while next_image:
            im1 = next_image
            next_image = None

            for im2 in per_image[im1]:
                if (im2, im1) not in ordered:
                    ordered.add((im1, im2))
                    if not next_image and im2 in remaining:
                        next_image = im2
                        remaining.remove(im2)

            if not next_image and remaining:
                next_image = remaining.pop()

    return list(ordered)
