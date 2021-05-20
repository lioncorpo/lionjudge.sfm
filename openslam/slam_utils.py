from opensfm import features
from opensfm import types
import numpy as np

def in_image(point, width, height):
    if width > height:
        factor = height / width
        return point[0] >= -0.5 and point[0] <= 0.5 and \
            point[1] >= factor * -0.5 and point[1] <= factor * 0.5
    # height >= width
    factor = width / height
    return point[1] >= -0.5 and point[1] <= 0.5 and \
        point[0] >= factor * -0.5 and point[0] <= factor * 0.5


def extract_features(image: str, data):
    p_unmasked, f_unmasked, c_unmasked =\
        features.extract_features(data.load_image(image), data.config)

    fmask = data.load_features_mask(image, p_unmasked)

    p_unsorted = p_unmasked[fmask]
    f_unsorted = f_unmasked[fmask]
    c_unsorted = c_unmasked[fmask]
    return (p_unsorted, f_unsorted, c_unsorted)


def extract_colors_for_pts(image, points):
    xs = points[:, 0].round().astype(int)
    ys = points[:, 1].round().astype(int)
    if len(image.shape) == 2:
        image = np.dstack((image, image, image))
    return image[ys, xs]


def mat_to_pose(mat):
    pose = types.Pose()
    pose.set_rotation_matrix(mat[0:3, 0:3])
    pose.translation = mat[0:3,3]
    return pose
