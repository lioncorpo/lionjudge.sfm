import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from slam_types import Frame


class SlamFeatureExtractor(object):
    """Extracts features on N pyramid levels and a particular grid size"""

    def __init__(self, config_slam):
        self.cell_size = config_slam['feat_cell_size']
        self.cell_overlap = config_slam['feat_cell_overlap']
        self.pyr_levels = config_slam['feat_pyr_levels']
        self.patch_radius = 19  # [px]
        scale_factor = config_slam['feat_scale']
        self.scale_factors = [1.0]
        for lvl in range(1, self.pyr_levels):
            self.scale_factors.\
                append(self.scale_factors[lvl - 1] * scale_factor)
        
        # Extractor (use FAST for now)
        # TODO: Make this configurable!
        self.detector_init =\
            cv2.FastFeatureDetector_create(config_slam['feat_fast_ini_th'], True)
        self.detector_min =\
            cv2.FastFeatureDetector_create(config_slam['feat_fast_min_th'], True)

    def compute_image_pyramid(self, frame: Frame):
        img_pyr = frame.img_pyr
        size = np.array(frame.image.shape)
        for lvl in range(1, self.pyr_levels):
            size_lvl = np.asarray(np.round(size / self.scale_factors[lvl]), dtype=int)
            img_pyr.append(cv2.resize(img_pyr[lvl - 1], tuple(size_lvl[::-1]), 0, 0, cv2.INTER_LINEAR))

    def compute_keypoints(self, frame):
        min_border_x = min_border_y = self.patch_radius
        # img_pyr = frame.img_pyr
        keypts = []
        for lvl in range(0, self.pyr_levels):
            img_lvl = frame.img_pyr[lvl]
            # TODO: This part can be easily pre-computed
            orig_size = img_lvl.shape
            max_border_x = orig_size[1] - self.patch_radius
            max_border_y = orig_size[0] - self.patch_radius
            width = max_border_x - min_border_x
            height = max_border_y - min_border_y
            n_cols = math.ceil(width / self.cell_size) + 1
            n_rows = math.ceil(height / self.cell_size) + 1
            print(n_cols, n_rows)
            kp_lvl = []
            # Now, iterate through the patches
            for row in range(0, n_rows):
                min_y = min_border_y + row * self.cell_size
                # boundary checks
                if max_border_y - self.cell_overlap <= min_y:
                    continue
                max_y = min(min_y + self.cell_size + self.cell_overlap, max_border_y)
                for col in range(0, n_cols):
                    min_x = min_border_x + col * self.cell_size
                    if max_border_x - self.cell_overlap <= min_x:
                        continue
                    max_x = min(min_x + self.cell_size + self.cell_overlap, max_border_x)
                    kp = self.compute_from_patch(img_lvl[min_y:max_y, min_x:max_x])
                    im_patch = cv2.drawKeypoints(img_lvl[min_y:max_y, min_x:max_x], kp, None, color=(255,0,0))
                    cv2.imwrite("./debug/patch_"+str(lvl)+"_"+str(col)+"_"+str(row)+".png", im_patch)
                    print("Found {} keypts in patch {}/{}".format(len(keypts), row, col))
                    if len(kp) == 0:
                        continue
                    # go through the keypoints and convert patch coord to image coord
                    for pt in kp:
                        pt.x += (col * self.cell_size)
                        pt.y += (row * self.cell_size)
                        kp_lvl.append(p)
            n_kpts_lvl = 1000 # TODO: compute
            borders = (min_border_x, max_border_x, min_border_y, max_border_y)
            kpts_level = self.dist_keypoints(kpts_level, borders, n_kpts_lvl)       
        return keypts
    
    def dist_keypoints(self, kpts_level, borders, n_kpts_lvl):
        

        pass

    def compute_from_patch(self, patch):
        keypts = self.detector_init.detect(patch)
        if len(keypts) == 0:  # No detection, try again
            keypts = self.detector_min.detect(patch)
        return keypts

        