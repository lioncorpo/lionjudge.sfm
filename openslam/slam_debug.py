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
                        im, camera, title="", obs_normalized=True, do_show=True):
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
    # im = data.load_image(image)
    # print("Show image ", image)
    if len(im.shape) == 3:
        h1, w1, c = im.shape
    else:
        h1, w1 = im.shape
    pt = features.denormalized_image_coordinates(points2D, w1, h1)
    # print("obs:", obs)
    # print("points2D: ", points2D)
    ax.imshow(im)
    ax.scatter(pt[:, 0], pt[:, 1], c=[[1, 0, 0]])
    if observations is not None:
        if obs_normalized:
            obs = features.denormalized_image_coordinates(observations, w1, h1)
        else:
            obs = observations
        ax.scatter(obs[:, 0], obs[:, 1], c=[[0, 1, 0]])
    ax.set_title(title)
    if do_show:
        plt.show()


def draw_observations_in_image(observations, image, data, do_show=True):
    """Draws observations into image"""
    if disable_debug:
        return
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


def draw_obs_in_image_no_norm(obs, image, title="observations", do_show=True):
    if disable_debug:
        return
    fig, ax = plt.subplots(1)
    if len(image.shape) == 3:
        # h1, w1, c = image.shape
        im = image
    else:
        im = np.dstack((image, image, image))
        # h1, w1, c = image.shape
    
    ax.imshow(im)
    ax.scatter(obs[:, 0], obs[:, 1], c=[[0, 1, 0]])
    ax.set_title(title)
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
    skip = 25
    ax.scatter(obs_d1[:, 0], obs_d1[:, 1], c=[[0, 1, 0]])
    ax.scatter(w1+obs_d2[:, 0], obs_d2[:, 1], c=[[0, 1, 0]])
    for a, b in zip(obs_d1[::skip, :], obs_d2[::skip, :]):
        ax.plot([a[0], b[0] + w1], [a[1], b[1]])
    ax.set_title(title)
    if do_show:
        plt.show()


def visualize_epipolar_line(pts1, pts2, im1, im2, T_1_2, K, min_d, max_d):
    # This is probably extremely slow but just for debug
    if len(im1.shape) == 3:
        h1, w1, c = im1.shape
    else:
        h1, w1 = im1.shape
    fig, ax = plt.subplots(1)
    im = np.vstack((im1, im2))
    obs_d1, obs_d2 = pts1, pts2
    ax.imshow(im)
    ax.scatter(obs_d1[:, 0], obs_d1[:, 1], c=[[0, 1, 0]])
    ax.scatter(obs_d2[:, 0], h1+obs_d2[:, 1], c=[[0, 1, 0]])
    K_inv = np.linalg.inv(K)
    n_steps = 20
    margin = 5
    KRK_i = K.dot(T_1_2[0:3, 0:3].dot(K_inv))
    Kt = K.dot(T_1_2[0:3, 3])
    # T_1_2 = np.linalg.inv(T_1_2)
    # Now, compute the epipolar line
    for idx, pt2 in enumerate(pts2[:, 0:2]):
        if idx % 50 == 0:
            pt3D = KRK_i.dot(np.hstack((pt2, 1)))
            start = pt3D*min_d + Kt # T_1_2[0:3, 0:3].dot(K_inv.dot(np.hstack((pt2, 1)) * min_d)) + T_1_2[0:3, 3]
            end = pt3D*max_d + Kt # T_1_2[0:3, 0:3].dot(K_inv.dot(np.hstack((pt2, 1))* max_d)) + T_1_2[0:3, 3]
            # end = K.dot(pt3D_h)
            start = start[0:2] / start[2]
            # end = K.dot(pt3D_h2)
            end = end[0:2] / end[2]
            plt.plot([start[0], end[0]], [start[1], end[1]], color='blue', linewidth=2)
            plt.plot([start[0], pt2[0]], [start[1], h1 + pt2[1]], color=np.random.rand(3), linewidth=2)

            # now iterate through the depth
            for factor in np.arange(0,5,0.4):
                start =pt3D * min_d*factor + Kt
                # pt3D_h2 = T_1_2[0:3, 0:3].dot(K_inv.dot(np.hstack((pt2, 1))* max_d)) + T_1_2[0:3, 3]
                # start = K.dot(pt3D_h)
                z = start[2]
                start = start[0:2] / start[2]
                plt.scatter(start[0],start[1],marker='x')
                plt.text(start[0],start[1],s=str(int(min_d*factor*100)/100.0)+"/"+str(int(z*100)/100.0), c='blue')
            
        continue
        # if idx % 15 == 0:
        #     pt3D_h = T_1_2[0:3,0:3].dot(K_inv.dot(np.hstack((pt2, 1))*depth))+T_1_2[0:3,3]
        #     pt3D_h2 = T_1_2[0:3,0:3].dot(K_inv.dot(np.hstack((pt2, 1))*(depth+0.1)))+T_1_2[0:3,3]
        #     start = K.dot(pt3D_h)
        #     start = start[0:2]/start[2]
        #     end = K.dot(pt3D_h2)
        #     end = end[0:2]/end[2]
        #     unit_epi = start-end
        #     unit_epi = unit_epi/np.linalg.norm(unit_epi)

        #     # draw the start
        #     e1 = start + unit_epi*20*margin*1.5
        #     e2 = start - unit_epi*20*margin*1.5
        #     ax.scatter(start[0], start[1], c=[[0,1,0]])
        #     plt.plot([e1[0], e2[0]], [e1[1], e2[1]], color='blue', linewidth=2)
        #     plt.plot([start[0], pt2[0]], [start[1], h1+pt2[1]], color='red', linewidth=2)
    plt.show()

    


def visualize_matches_pts(pts1, pts2, matches, im1, im2, is_normalized= True, do_show=True, title = ""):
    if disable_debug:
        return
    
    if len(im1.shape) == 3:
        h1, w1, c = im1.shape
    else:
        h1, w1 = im1.shape
    fig, ax = plt.subplots(1)
    im = np.hstack((im1, im2))
    if is_normalized:
        obs_d1 = features.\
            denormalized_image_coordinates(np.asarray(pts1[matches[:, 0]]), w1, h1)
        obs_d2 = features.\
            denormalized_image_coordinates(np.asarray(pts2[matches[:, 1]]), w1, h1)
    else:
        obs_d1, obs_d2 = pts1[matches[:, 0]], pts2[matches[:, 1]]
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
    if disable_debug:
        return
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
    # if disable_debug:
        # return
    im1 = data.load_image(frame.im_name)
    h1, w1, c = im1.shape
    im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2BGR)
    p1d = np.array(features.denormalized_image_coordinates(points2D, w1, h1), dtype=int)
    for x, y in p1d:
        cv2.drawMarker(im1, (x, y), (255, 0, 0),
                       markerType=cv2.MARKER_SQUARE, markerSize=10)
    
    cv2.imwrite("./debug/track_"+frame.im_name, im1)
    # cv2.imwrite("/home/fschenk/software/mapillary_repos/OpenSfM/debug/track_"+frame.im_name, im1)
