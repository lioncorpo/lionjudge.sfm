import os.path, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from openslam import initializer
from slam_types import Frame
from slam_system import SlamSystem
import slam_debug
import argparse
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)
import cv2
from opensfm import dataset
from opensfm import features
from opensfm import io

# import cslam_types

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Create the top-level parser
parser = argparse.ArgumentParser()
parser.add_argument('dataset', help='dataset to process')
args = parser.parse_args()
args.dataset = "/home/fschenk/software/mapillary_repos/mapillary_sfm_evaluation/sfm_evaluation_workspace/kitti_00"
# args.dataset = "/home/fschenk/data/ae_sequences/single_images/blackvue20190820_235022_NF"
slam_system = SlamSystem(args)
data = dataset.DataSet(args.dataset)
start_id = 0
for idx, im_name in enumerate(sorted(data.image_list)):
    if idx < start_id:
        continue
    gray_scale_img = io.imread(data._image_file(im_name), grayscale=True)  # The gray-scale image
    ret = slam_system.process_frame_2(im_name, gray_scale_img)
    slam_debug.avg_timings.printAvgTimings()
    if ret:
        logger.info("Successfully tracked {}".format(im_name))
        if idx == 4:
            exit()
    else:
        logger.info("Trying to init with {}".format(im_name))


slam_system.slam_mapper.create_reconstruction()
slam_system.slam_mapper.save_reconstruction(im_name + "_finished")
