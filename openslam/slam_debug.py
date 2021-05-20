import matplotlib.pyplot as plt
from opensfm import features
from slam_types import Frame
import cv2
import numpy as np
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

disable_debug = True


class AvgTimings(object):
    def __init__(self):
        self.times = defaultdict(float)
        self.n_mean = defaultdict(int)

    def addTimes(self, timings):
        for (_, (k, v, _)) in timings.items():
            self.times[k] += v
            self.n_mean[k] += 1

    def printAvgTimings(self):
        for (k, v) in self.n_mean.items():
            print("{} with {} runs: {}s".format(k, v, self.times[k]/v))


avg_timings = AvgTimings()


def reproject_landmarks(points3D, observations, pose_world_to_cam,
                        image, camera, data, title="", do_show=True):
    """Draw observations and reprojects observations into image"""
    if disable_debug:
        return

    if points3D is None: #or observations is None:
        return
    if len(points3D) == 0: #or len(observations) == 0:
        return
    camera_point = pose_world_to_cam.transform_many(points3D)
    points2D = camera.project_many(camera_point)
    fig, ax = plt.subplots(1)
    im = data.load_image(image)
    print("Show image ", image)
    h1, w1, c = im.shape
    pt = features.denormalized_image_coordinates(points2D, w1, h1)
    # print("obs:", obs)
    # print("points2D: ", points2D)
    ax.imshow(im)
    ax.scatter(pt[:, 0], pt[:, 1], c=[[1, 0, 0]])
    if observations is not None:
        obs = features.denormalized_image_coordinates(observations, w1, h1)
        ax.scatter(obs[:, 0], obs[:, 1], c=[[0, 1, 0]])
    ax.set_title(title)
    if do_show:
        plt.show()


def draw_observations_in_image(observations, image, data, do_show=True):
    """Draws observations into image"""
    if disable_debug:
        return
    if observations is None:
        return
    if len(observations) == 0:
        return
    fig, ax = plt.subplots(1)
    im = data.load_image(image)
    h1, w1, c = im.shape
    obs = features.denormalized_image_coordinates(observations, w1, h1)
    ax.imshow(im)
    ax.scatter(obs[:, 0], obs[:, 1], c=[[0, 1, 0]])
    ax.set_title(image)
    if do_show:
        plt.show()


def visualize_matches_pts(pts1, pts2, matches, im1, im2, do_show=True, title = ""):
    if disable_debug:
        return
    h1, w1, c = im1.shape
    fig, ax = plt.subplots(1)
    im = np.hstack((im1, im2))
    obs_d1 = features.\
        denormalized_image_coordinates(np.asarray(pts1[matches[:, 0]]), w1, h1)
    obs_d2 = features.\
        denormalized_image_coordinates(np.asarray(pts2[matches[:, 1]]), w1, h1)
    ax.imshow(im)
    skip = 5
    ax.scatter(obs_d1[:, 0], obs_d1[:, 1], c=[[0, 1, 0]])
    ax.scatter(w1+obs_d2[:, 0], obs_d2[:, 1], c=[[0, 1, 0]])
    for a, b in zip(obs_d1[::skip, :], obs_d2[::skip, :]):
        ax.plot([a[0], b[0] + w1], [a[1], b[1]])
    ax.set_title(title)
    if do_show:
        plt.show()

def visualize_matches(matches, frame1: str, frame2: str, data, do_show=True):
    if disable_debug:
        return
    im1 = data.load_image(frame1)
    im2 = data.load_image(frame2)
    h1, w1, c = im1.shape
    fig, ax = plt.subplots(1)
    im = np.hstack((im1, im2))
    p1, _, _ = data.load_features(frame1)
    p2, _, _ = data.load_features(frame2)
    pts2D_1 = p1[matches[:, 0], 0:2]
    pts2D_2 = p2[matches[:, 1], 0:2]
    obs_d1 = features.\
        denormalized_image_coordinates(np.asarray(pts2D_1), w1, h1)
    obs_d2 = features.\
        denormalized_image_coordinates(np.asarray(pts2D_2), w1, h1)
    ax.imshow(im)
    ax.scatter(obs_d1[:, 0], obs_d1[:, 1], c=[[0, 1, 0]])
    ax.scatter(w1+obs_d2[:, 0], obs_d2[:, 1], c=[[0, 1, 0]])
    for a, b in zip(obs_d1[::10, :], obs_d2[::10, :]):
        ax.plot([a[0], b[0] + w1], [a[1], b[1]])
    ax.set_title(frame1 + "<->" + frame2)
    if do_show:
        plt.show()


def visualize_graph(graph, frame1: str, frame2: str, data, do_show=True):
    print("visualize_graph: ", frame1, frame2)
    lms = graph[frame1]
    pts2D_1 = []
    pts2D_2 = []
    for lm_id in lms:
        obs2 = \
            graph.get_edge_data(str(frame2), str(lm_id))
        if obs2 is not None:
            obs1 = \
                graph.get_edge_data(str(frame1), str(lm_id))
            pts2D_1.append(obs1['feature'])
            pts2D_2.append(obs2['feature'])
    if len(pts2D_1) == 0:
        return
    im1 = data.load_image(frame1)
    im2 = data.load_image(frame2)
    h1, w1, c = im1.shape
    fig, ax = plt.subplots(1)

    obs_d1 = features.\
        denormalized_image_coordinates(np.asarray(pts2D_1), w1, h1)
    obs_d2 = features.\
        denormalized_image_coordinates(np.asarray(pts2D_2), w1, h1)
    print("len(obs_d1): ", len(obs_d1), "len(obs_d2): ", len(obs_d2))
    im = np.hstack((im1, im2))
    ax.imshow(im)
    ax.scatter(obs_d1[:, 0], obs_d1[:, 1], c=[[0, 1, 0]])
    ax.scatter(w1+obs_d2[:, 0], obs_d2[:, 1], c=[[0, 1, 0]])
    ax.set_title(frame1 + "<->" + frame2)

    if do_show:
        plt.show()


def visualize_tracked_lms(points2D, frame: Frame, data):
    im1 = data.load_image(frame.im_name)
    h1, w1, c = im1.shape
    im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2BGR)
    p1d = np.array(features.denormalized_image_coordinates(points2D, w1, h1), dtype=int)
    for x, y in p1d:
        cv2.drawMarker(im1, (x, y), (255, 0, 0),
                       markerType=cv2.MARKER_SQUARE, markerSize=10)
    
    cv2.imwrite("./debug/track_"+frame.im_name, im1)
    # cv2.imwrite("/home/fschenk/software/mapillary_repos/OpenSfM/debug/track_"+frame.im_name, im1)
