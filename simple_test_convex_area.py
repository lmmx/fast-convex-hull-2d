from sys import stderr
import numpy as np
from skimage.measure import regionprops
from skimage.morphology.convex_hull import _offsets_diamond
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

def common_subroutine_2(coords, no_val_dict_update=False):
    hull = ConvexHull(coords)
    vertices = hull.points[hull.vertices]
    hull_perim_r, hull_perim_c = polygon_perimeter(vertices[:, 0], vertices[:, 1])
    mask = np.zeros(img.shape, dtype=np.bool)
    if not no_val_dict_update:
        for val in [*vars()]:
            VAL_DICT.setdefault(val, [])
            VAL_DICT[val].append(vars()[val])
    return hull, vertices, hull_perim_r, hull_perim_c, mask

def nobug(img, coords):
    hull, vertices, hull_perim_r, hull_perim_c, mask = common_subroutine_2(coords)
    mask[hull_perim_r, hull_perim_c] = True # no IndexError here!

def bug(img, coords, rets=False):
    # Now include the intermediate processing steps where offsets are applied...
    offsets = _offsets_diamond(img.ndim)
    coords = (coords[:, np.newaxis, :] + offsets).reshape(-1, img.ndim)
    # ...skip some more code as "the damage has been done" by the function above
    hull, vertices, hull_perim_r, hull_perim_c, mask = common_subroutine_2(coords,rets)
    if rets: # Return early (otherwise cannot return the intermediate values)
        return offsets, coords, hull, vertices, hull_perim_r, hull_perim_c, mask
    mask[hull_perim_r, hull_perim_c] = True # raises IndexError

rp = regionprops(SAMPLE)[0]
img = SAMPLE.astype(np.bool)
bug_list = [nobug, bug]
for bugfunc in bug_list:
    img, coords = common_subroutine_1(img=img)
    pre_coords = coords # store these as they change
    try:
        bugfunc(img, coords)
        print(f"{bugfunc.__name__} raised no error.", file=stderr)
    except IndexError as e:
        print(f"{bugfunc.__name__} raised the IndexError {e}", file=stderr)
# Populate the namespace with the resulting variables of `bug`
rets = bug(img, coords, rets=True)
offsets, coords, hull, vertices, hull_perim_r, hull_perim_c, mask = rets
