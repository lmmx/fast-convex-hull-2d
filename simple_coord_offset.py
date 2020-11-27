from sys import stderr
import numpy as np
from skimage.measure import regionprops
from skimage.morphology.convex_hull import _offsets_diamond
from sample_data import SAMPLE
from common_subroutines import common_subroutine_1, common_subroutine_2

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

def bugfix_new(img, coords, rets=False):
    # Now include the intermediate processing steps where offsets are applied...
    offsets = _offsets_diamond(img.ndim)
    coords = (coords[:, np.newaxis, :] + offsets).reshape(-1, img.ndim)
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
    pre_coords = coords # store these as they change
    try:
        bugfunc(img, coords)
        print(f"{bugfunc.__name__} raised no error.", file=stderr)
    except IndexError as e:
        print(f"{bugfunc.__name__} raised the IndexError {e}", file=stderr)
# Populate the namespace with the resulting variables of `bugfix_draft`
rets = bugfix_draft(img, coords, rets=True)
offsets, coords, hull, vertices, hull_perim_r, hull_perim_c, mask = rets
