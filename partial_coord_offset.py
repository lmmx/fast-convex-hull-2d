import numpy as np
from skimage.measure import regionprops
from skimage.morphology.convex_hull import _offsets_diamond
from sample_data import SAMPLE
from skimage.morphology import _convex_hull
from skimage.draw import polygon_perimeter
from scipy.spatial import ConvexHull

def apply_partial_offsets(img, coords, offsets, retain_original_points=True):
    """
    Apply the offsets only to the non-edge pixels, along with the trivial zero-offset
    if `retain_original_points` is True (default: True, recommended due to corner loss).
    """
    if retain_original_points:
        # Insert the trivial offset of [0., 0.] into `offsets`
        offsets = np.insert(offsets, 0, 0., axis=0)
    row_max, col_max = np.subtract(img.shape, 1)
    # bool masks for the subsets of `coords` including each edge (one edge at a time)
    edge_t, edge_b = [coords[:,0] == lim for lim in (0, row_max)]
    edge_l, edge_r = [coords[:,1] == lim for lim in (0, col_max)]
    edge_includers = [edge_t, edge_b, edge_l, edge_r]
    if retain_original_points:
        dummy_edge = np.zeros_like(edge_t, dtype=bool) # False so offset always applied
        edge_includers.insert(0, dummy_edge)
    offset_mask = np.invert(edge_includers).T
    offset_idx = np.argwhere(offset_mask.ravel()).ravel()
    coords = (coords[:, np.newaxis, :] + offsets).reshape(-1, img.ndim)[offset_idx]
    return coords

rp = regionprops(SAMPLE)[0]
img = SAMPLE.astype(np.bool)
coords = _convex_hull.possible_hull(np.ascontiguousarray(img, dtype=np.uint8))
offsets = _offsets_diamond(img.ndim)
# Apply bugfix to apply these offsets in a partial (valid/limited) way to the coords
coords = apply_partial_offsets(img, coords, offsets)
hull = ConvexHull(coords)
vertices = hull.points[hull.vertices]
hull_perim_r, hull_perim_c = polygon_perimeter(vertices[:, 0], vertices[:, 1])
mask = np.zeros(img.shape, dtype=np.bool)
mask[hull_perim_r, hull_perim_c] = True
