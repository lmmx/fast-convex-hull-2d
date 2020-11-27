import numpy as np
from skimage.morphology import _convex_hull
possible_hull = _convex_hull.possible_hull # Not sure how else to do this
from scipy.spatial import ConvexHull
from skimage.draw import polygon_perimeter
from sample_data import SAMPLE

global VAL_DICT
VAL_DICT = {}

def common_subroutine_1(img=SAMPLE.astype(np.bool)):
    coords = possible_hull(np.ascontiguousarray(img, dtype=np.uint8))
    for val in [*vars()]:
        VAL_DICT.setdefault(val, [])
        VAL_DICT[val].append(vars()[val])
    return img, coords

def common_subroutine_2(img_shape, coords, no_val_dict_update=False):
    hull = ConvexHull(coords)
    vertices = hull.points[hull.vertices]
    hull_perim_r, hull_perim_c = polygon_perimeter(vertices[:, 0], vertices[:, 1])
    mask = np.zeros(img_shape, dtype=np.bool)
    if not no_val_dict_update:
        for val in [*vars()]:
            VAL_DICT.setdefault(val, [])
            VAL_DICT[val].append(vars()[val])
    return hull, vertices, hull_perim_r, hull_perim_c, mask
