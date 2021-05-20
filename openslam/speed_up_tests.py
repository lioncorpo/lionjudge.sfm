import cv2
from opensfm import reconstruction
from opensfm import dataset
from opensfm import feature_loader
from opensfm import features
from opensfm import matching
from openslam import slam_debug
import numpy as np
#load two images
data = dataset.DataSet("/home/fschenk/data/ae_sequences/single_images/blackvue20190820_235022_NF")
images = sorted(data.image_list)
im1 = data.load_image(images[0])
im2 = data.load_image(images[1])
im1g = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
im2g = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)

chrono = reconstruction.Chronometer()
n_test = 1
# test feature detectors
# Harris
# Detector parameters
blockSize = 2
apertureSize = 3
k = 0.04
# Detecting corners
for i in range(0,n_test):
      corners_harris = cv2.cornerHarris(im1g, blockSize, apertureSize, k)
chrono.lap("harris detect")
# ORB
orb = cv2.ORB_create(nfeatures=int(4000))
chrono.lap("orb create")
for i in range(0,n_test):
      points = orb.detect(im1g)
chrono.lap("orb detect")
for i in range(0,n_test):
      kpts, points = orb.detectAndCompute(im1g, None)
chrono.lap("orb detect+comp")
print(chrono.lap_times())
# exit(0)
for i in range(0,n_test):
      points = features.extract_features_orb(im1g, data.config)
chrono.lap("orb detect opensfm")
# Fast
fast = cv2.FastFeatureDetector_create()
# fast.setNonmaxSuppression(0)
chrono.lap("fast create")
for i in range(0,n_test):
      kp = fast.detect(im1g, None)
chrono.lap("fast detect")


chrono.lap("det ahog")
for i in range(0,n_test):
      features.extract_features_hahog(im1g, data.config)
chrono.lap("det ahog fin")
# print(chrono.laps_dict['fast detect'][1]/n_test)
# print(chrono.lap_times())
# exit(0)
#godd features to track
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 200,
                       qualityLevel = 0.3,
                       minDistance = 20,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (7,7),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0, 255, (100,3))
p0 = cv2.goodFeaturesToTrack(im1g, mask = None, **feature_params, useHarrisDetector=True)
chrono.lap('good features harrus')
p0 = cv2.goodFeaturesToTrack(im1g, mask = None, **feature_params, useHarrisDetector=False)
chrono.lap('good features tomasi')
print("Times: ", chrono.lap_times())

# load the desc + pts
p1, f1, c1 = data.load_features(images[0])
p2, f2, c2 = data.load_features(images[1])
print("orb: ", len(points), " fast: ", len(kp), " harris: ",
      len(corners_harris), "good: ", len(p0), "hog:", len(p1))
# print(p1, p0)
cam = next(iter(data.load_camera_models_overrides().values()))
p1_lk = features.denormalized_image_coordinates(p1, cam.width, cam.height)
p1_lk = p0
print("cam: ", cam)

# Parameters for lucas kanade optical flow
# print(np.reshape(p1_lk,[-1,1,2]), p1_lk.dtype)

chrono = reconstruction.Chronometer()
# test LK tracker vs matching
# LK
p1_lk = np.asarray(p1_lk.reshape([-1, 1, 2]), dtype=np.float32)
# calculate optical flow
p2_lk, st, err = cv2.calcOpticalFlowPyrLK(im1g, im2g, p1_lk, None, **lk_params)
# pyr1 = None
nLvl, pyr1 = cv2.buildOpticalFlowPyramid(im1g, lk_params['winSize'], lk_params['maxLevel'], withDerivatives=False)
print(nLvl, pyr1)
# for p in pyr1:

nLvl2, pyr2 = cv2.buildOpticalFlowPyramid(im2g, lk_params['winSize'], lk_params['maxLevel'], withDerivatives=False)
pyr1_lk = []
for p in pyr1:
      pyr1_lk.append(np.float32(p))
      print("p: ", p.shape)

# _, _, _ = cv2.calcOpticalFlowPyrLK((np.array(nLvl, np.array(pyr1))), np.array((nLvl2, np.array(pyr2))), p1_lk, None, **lk_params)
# p1_lk = []
_, _, _ = cv2.calcOpticalFlowPyrLK((pyr1[0],pyr1[1]), (pyr1[0],pyr1[1]), p1_lk, None, **lk_params)
_, _, _ = cv2.calcOpticalFlowPyrLK((nLvl, pyr1_lk), (nLvl2, pyr1_lk), p1_lk, None, **lk_params)
_, _, _ = cv2.calcOpticalFlowPyrLK(cv2.UMat(pyr1), cv2.UMat(pyr2), p1_lk, None, **lk_params)
_, _, _ = cv2.calcOpticalFlowPyrLK((nLvl, np.asarray(pyr1)), (nLvl2, np.asarray(pyr2)), p1_lk, None, **lk_params)
_, _, _ = cv2.calcOpticalFlowPyrLK((nLvl, pyr1), (nLvl2, pyr2), p1_lk, None, **lk_params)
# print(pyr1)
# p2_lk, st, err = cv2.calcOpticalFlowPyrLK(im1g, im2g, np.reshape(p1_lk,[-1,1,2]), None, **lk_params)
chrono.lap('lk_match')
st = st == 1
# Pyramid
p1_lk = p1_lk[st == 1]
p2_lk = p2_lk[st == 1]
a = np.arange(0, len(p1_lk))
matches = np.vstack((a, a)).transpose()
chrono.lap('lk_filter')
p1_lk_n = features.normalized_image_coordinates(p1_lk, cam.width, cam.height)
p2_lk_n = features.normalized_image_coordinates(p2_lk, cam.width, cam.height)
chrono.lap('lk norm')
rmatches = matching.robust_match(p1_lk_n, p2_lk_n, cam, cam, matches, data.config)
rmatches = np.array([[a, b] for a, b in rmatches])
chrono.lap('lk robust match')
print(p1_lk.shape, p2_lk.shape)
# out = []
# cv2.drawMatches(im1, p1_lk, im2, p2_lk, rmatches, out)
slam_debug.visualize_matches_pts(p1_lk_n, p2_lk_n, rmatches, im1, im2, False)
# matching
print(chrono.lap_times())
print("lk matches: ", np.sum(st), " rmatches: ", len(rmatches))
chrono = reconstruction.Chronometer()
# matching descriptors
i1 = feature_loader.instance.load_features_index(data, images[0], masked=True)
i2 = feature_loader.instance.load_features_index(data, images[1], masked=True)
chrono.lap('load index')
matches = matching.match_flann_symmetric(f1, i1, f2, i2, data.config)
matches = np.asarray(matches, dtype=int)
chrono.lap('match_flann')
rmatches = matching.robust_match(p1, p2, cam, cam, matches, data.config)
rmatches = np.array([[a, b] for a, b in rmatches])
chrono.lap('robust match')
print(chrono.lap_times())
print("len(matches): ", len(matches), "len(rmatches):", len(rmatches))
slam_debug.visualize_matches_pts(p1, p2, rmatches, im1, im2, True)