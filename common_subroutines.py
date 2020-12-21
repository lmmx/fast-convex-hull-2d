import numpy as np
from skimage.morphology import _convex_hull
from scipy.spatial import ConvexHull
from skimage.draw import polygon_perimeter
from sample_data import SAMPLE

def get_possible_hull(img):
    coords = _convex_hull.possible_hull(np.ascontiguousarray(img, dtype=np.uint8))
    return img, coords

def get_mask_and_hull(img_shape, coords):
    hull = ConvexHull(coords)
    vertices = hull.points[hull.vertices]
    hull_perim_r, hull_perim_c = polygon_perimeter(vertices[:, 0], vertices[:, 1])
    mask = np.zeros(img_shape, dtype=np.bool)
    return mask, hull, vertices, hull_perim_r, hull_perim_c
