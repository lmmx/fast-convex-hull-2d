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
    return mask

def apply_partial_offsets(img_shape, coords, offsets):
    "Apply the offsets only to the non-edge pixels"
    # The following line is what we're replacing with the more complicated procedure below
    #coords = (coords[:, np.newaxis, :] + offsets).reshape(-1, img.ndim)
    row_min, col_min = 0, 0
    row_max, col_max = np.subtract(img_shape, 1)
    # bool masks for the subsets of `coords` excluding each edge (one edge at a time)
    not_t, not_b = [coords[:,0] != lim for lim in (row_min, row_max)]
    not_l, not_r = [coords[:,1] != lim for lim in (col_min, col_max)]
    edge_excluders = [not_t, not_b, not_l, not_r]
    # intersection (AND) of all edge excluders, i.e. the bool mask for non-edge coords
    not_edge = reduce(np.logical_and, edge_excluders)
    # bool masks for the subsets of `coords` including each edge (one edge at a time)
    edge_t, edge_b = [coords[:,0] == lim for lim in (row_min, row_max)]
    edge_l, edge_r = [coords[:,1] == lim for lim in (col_min, col_max)]
    edge_includers = edge_t, edge_b, edge_l, edge_r
    offset_mask = np.invert(edge_includers).T
    coords_t, coords_b, coords_l, coords_r = [coords[e] for e in edge_includers]
    # union (OR) of all the edge includers, i.e. the bool mask for edge coords
    any_edge = reduce(np.logical_or, edge_includers)
    any_edge_i = np.invert(not_edge) # just a sanity check, this is unnecessary
    assert np.array_equal(any_edge, any_edge_i), "Edge includer not inverse of excluder"
    inner_coords = coords[not_edge]
    # inner_coords can be safely treated with all offsets
    inner_coords = (inner_coords[:, np.newaxis, :] + offsets).reshape(-1, img.ndim)
    offsets_b, offsets_t, offsets_r, offsets_l = offsets
    # edge coords must be treated one at a time (can this be vectorised i.e. in 1 line?)
    coords_t = (coords_t[:, np.newaxis, :] + offsets_t).reshape(-1, img.ndim)
    coords_b = (coords_b[:, np.newaxis, :] + offsets_b).reshape(-1, img.ndim)
    coords_l = (coords_l[:, np.newaxis, :] + offsets_l).reshape(-1, img.ndim)
    coords_r = (coords_r[:, np.newaxis, :] + offsets_r).reshape(-1, img.ndim)
    edge_coords = np.unique(np.vstack([coords_t, coords_r, coords_b, coords_l]), axis=0)
    # form new coords array by concatenating the inner and the edge offsets
    coords = np.vstack([inner_coords, edge_coords])
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
    return mask

rp = regionprops(SAMPLE)[0]
img = SAMPLE.astype(np.bool)
bug_list = [bugfix_draft, bugfix_new]
final_masks = []
for bugfunc in bug_list:
    img, coords = common_subroutine_1(img=img)
    pre_coords = coords.copy() # store these as they change
    try:
        mask = bugfunc(img, coords)
        print(f"{bugfunc.__name__} raised no error.", file=stderr)
        final_masks.append(mask)
    except IndexError as e:
        print(f"{bugfunc.__name__} raised the IndexError {e}", file=stderr)
# Populate the namespace with the resulting variables of `bugfix_draft`
rets = bugfix_draft(img, coords, rets=True)
offsets_good, coords_good, hull_good, vertices_good, hull_perim_r_good, hull_perim_c_good, mask_good = rets
#
rets = bugfix_new(img, coords, rets=True)
offsets, coords, hull, vertices, hull_perim_r, hull_perim_c, mask = rets

is_same_result = np.array_equal(*final_masks)
print(f"Bug fixes give same result: {is_same_result}")
