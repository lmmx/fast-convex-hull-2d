from sys import stderr
import numpy as np
from skimage.measure import regionprops
from skimage.morphology.convex_hull import _offsets_diamond
from sample_data import SAMPLE
from common_subroutines import common_subroutine_1, common_subroutine_2
from functools import reduce

global VAL_DICT
VAL_DICT = {}

def bugfix_draft(img, coords, rets=False):
    # Now include the intermediate processing steps where offsets are applied...
    offsets = _offsets_diamond(img.ndim)
    coords = (coords[:, np.newaxis, :] + offsets).reshape(-1, img.ndim)
    # But remediate the out-of-bounds indices
    coords[coords[:,0] == coords[:,0].min()] += [0.5, 0.]
    coords[coords[:,1] == coords[:,1].min()] += [0., 0.5]
    coords[coords[:,0] == coords[:,0].max()] -= [0.5, 0.]
    coords[coords[:,1] == coords[:,1].max()] -= [0., 0.5]
    # ...skip some more code as "the damage has been done" by the function above
    hull, vertices, hull_perim_r, hull_perim_c, mask = common_subroutine_2(img.shape,coords,rets)
    if rets: # Return early (otherwise cannot return the intermediate values)
        return offsets, coords, hull, vertices, hull_perim_r, hull_perim_c, mask
    mask[hull_perim_r, hull_perim_c] = True # raises IndexError

def apply_partial_offsets(img_shape, coords, offsets):
    "Apply the offsets only to the non-edge pixels"
    coords = (coords[:, np.newaxis, :] + offsets).reshape(-1, img.ndim)
    row_min, col_min = 0, 0
    row_max, col_max = np.subtract(img_shape, 1)
    not_t = coords[:,0] != row_min
    not_b = coords[:,0] != row_max
    not_l = coords[:,1] != col_min
    not_r = coords[:,1] != col_max
    edge_excluders = [not_t, not_b, not_l, not_r]
    not_edge = reduce(np.logical_and, edge_excluders)
    exc_coords = coords[edges]
    exc_r, exc_c = exc_coords[:,0], exc_coords[:,1]
    img_exc = edit_img[exc_r, exc_c]
    mid_indx = np.argwhere(np.logical_and(0 < img_exc, img_exc < 255)).reshape(-1)
    max_indx = np.argwhere(img_exc == 255).reshape(-1)
    mid_r, mid_c = exc_r[mid_indx], exc_c[mid_indx]
    max_r, max_c = exc_r[max_indx], exc_c[max_indx]
    edit_img[mid_r, mid_c] += 140
    edit_img[max_r, max_c] = 60
    frames.append(edit_img.copy())
    include = np.invert(not_edge)
    include_coords =
    coords[np.argwhere(include).reshape(-1)]
    return coords

def bugfix_new(img, coords, rets=False):
    # Now include the intermediate processing steps where offsets are applied...
    offsets = _offsets_diamond(img.ndim)
    #coords = (coords[:, np.newaxis, :] + offsets).reshape(-1, img.ndim)
    coords = apply_partial_offsets(img.shape, coords, offsets)
    # ...skip some more code as "the damage has been done" by the function above
    hull, vertices, hull_perim_r, hull_perim_c, mask = common_subroutine_2(img.shape,coords,rets)
    if rets: # Return early (otherwise cannot return the intermediate values)
        return offsets, coords, hull, vertices, hull_perim_r, hull_perim_c, mask
    mask[hull_perim_r, hull_perim_c] = True # raises IndexError

rp = regionprops(SAMPLE)[0]
img = SAMPLE.astype(np.bool)
bug_list = [bugfix_draft]
for bugfunc in bug_list:
    img, coords = common_subroutine_1(img=img)
    pre_coords = coords.copy() # store these as they change
    try:
        bugfunc(img, coords)
        print(f"{bugfunc.__name__} raised no error.", file=stderr)
    except IndexError as e:
        print(f"{bugfunc.__name__} raised the IndexError {e}", file=stderr)
# Populate the namespace with the resulting variables of `bugfix_draft`
rets = bugfix_draft(img, coords, rets=True)
offsets, coords, hull, vertices, hull_perim_r, hull_perim_c, mask = rets
