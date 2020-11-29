from sys import stderr
import numpy as np
from skimage.measure import regionprops
from skimage.morphology.convex_hull import _offsets_diamond
from sample_data import SAMPLE
from common_subroutines import common_subroutine_1, common_subroutine_2

global VAL_DICT
VAL_DICT = {}

def apply_partial_offsets(img, coords, offsets):
    "Apply the offsets only to the non-edge pixels, along with the trivial zero-offset."
    # The following line is what we're replacing with the more complicated procedure below
    #coords = (coords[:, np.newaxis, :] + offsets).reshape(-1, img.ndim)
    # Insert the trivial offset of [0., 0.] into `offsets`
    offsets = np.insert(offsets, 0, 0., axis=0)
    row_max, col_max = np.subtract(img.shape, 1)
    # bool masks for the subsets of `coords` including each edge (one edge at a time)
    edge_t, edge_b = [coords[:,0] == lim for lim in (0, row_max)]
    edge_l, edge_r = [coords[:,1] == lim for lim in (0, col_max)]
    dummy_edge = np.zeros_like(edge_t, dtype=bool) # all False so offset always applied
    edge_includers = dummy_edge, edge_t, edge_b, edge_l, edge_r
    offset_mask = np.invert(edge_includers).T
    offset_idx = np.argwhere(offset_mask.ravel()).ravel()
    coords = (coords[:, np.newaxis, :] + offsets).reshape(-1, img.ndim)[offset_idx]
    return coords

def bugfix(img, coords, rets=False):
    # Now include the intermediate processing steps where offsets are applied...
    offsets = _offsets_diamond(img.ndim)
    #coords = (coords[:, np.newaxis, :] + offsets).reshape(-1, img.ndim)
    coords = apply_partial_offsets(img, coords, offsets)
    # ...skip some more code as "the damage has been done" by the function above
    hull, vertices, hull_perim_r, hull_perim_c, mask = common_subroutine_2(img.shape,coords,rets)
    if rets: # Return early (otherwise cannot return the intermediate values)
        return offsets, coords, hull, vertices, hull_perim_r, hull_perim_c
    mask[hull_perim_r, hull_perim_c] = True # raises IndexError
    return mask

rp = regionprops(SAMPLE)[0]
img = SAMPLE.astype(np.bool)
img, coords = common_subroutine_1(img=img)
pre_coords = coords.copy() # store these as they change
try:
    mask = bugfix(img, coords)
    print(f"bugfix raised no error.", file=stderr)
except IndexError as e:
    print(f"{bugfunc.__name__} raised the IndexError {e}", file=stderr)

# Populate the namespace with the resulting variables of `bugfix`
rets = bugfix(img, coords, rets=True)
offsets, coords, hull, vertices, hull_perim_r, hull_perim_c = rets
