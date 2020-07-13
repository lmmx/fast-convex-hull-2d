from sys import stderr
import numpy as np
from skimage.measure import regionprops
from skimage.morphology.convex_hull import convex_hull_image, _offsets_diamond
from skimage.morphology import _convex_hull
possible_hull = _convex_hull.possible_hull # Not sure how else to do this
from scipy.spatial import ConvexHull
from skimage.draw import polygon_perimeter

SAMPLE = np.array(
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
     [1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
     [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1],
     [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]
)

def bug_initial():
    rp = regionprops(SAMPLE)[0]
    # sums `rp.convex_image` but accessing `convex_image` raises an IndexError
    area = rp.convex_area
    return

def bug_secondary():
    rp = regionprops(SAMPLE)[0]
    # accessing `convex_image` raises an IndexError
    conv_img = rp.convex_image
    return

def bug_tertiary():
    rp = regionprops(SAMPLE)[0]
    img = rp.image
    conv_img = convex_hull_image(img)
    return

def bug_quaternary():
    img = SAMPLE.astype(np.bool)
    conv_img = convex_hull_image(img)
    return

def nobug_5ary(img=None, rets=False):
    "If `rets` is True, return all variables to populate the caller's namespace"
    if img is None:
        img = SAMPLE.astype(np.bool)
    coords = possible_hull(np.ascontiguousarray(img, dtype=np.uint8))
    ##### (Skip the intermediate processing steps where offsets are applied) #####
    hull = ConvexHull(coords)
    vertices = hull.points[hull.vertices]
    hull_perim_r, hull_perim_c = polygon_perimeter(vertices[:, 0], vertices[:, 1])
    mask = np.zeros(img.shape, dtype=np.bool)
    # This is the line that reportedly raises IndexError but it doesn't here
    mask[hull_perim_r, hull_perim_c] = True
    if rets:
        return img, coords, hull, vertices, hull_perim_r, hull_perim_c, mask
    else:
        return

def bug_5ary(img=None, rets=False):
    "If `rets` is True, return all variables to populate the caller's namespace"
    if img is None:
        img = SAMPLE.astype(np.bool)
    coords = possible_hull(np.ascontiguousarray(img, dtype=np.uint8))
    # Now include the intermediate processing steps where offsets are applied...
    ndim = img.ndim
    offsets = _offsets_diamond(img.ndim)
    coords = (coords[:, np.newaxis, :] + offsets).reshape(-1, ndim)
    # ...skip some more code as "the damage has been done" by the function above
    # ...then continue with the rest of the func, which didn't error in `nobug_5ary`
    hull = ConvexHull(coords)
    vertices = hull.points[hull.vertices]
    hull_perim_r, hull_perim_c = polygon_perimeter(vertices[:, 0], vertices[:, 1])
    mask = np.zeros(img.shape, dtype=np.bool)
    if rets: # Return early (otherwise cannot return the intermediate values)
        return img, ndim, offsets, coords, hull, vertices, hull_perim_r, hull_perim_c, mask
    # This is the line that now raises IndexError
    mask[hull_perim_r, hull_perim_c] = True

if __name__ == "__main__":
    rp = regionprops(SAMPLE)[0]
    img = SAMPLE.astype(np.bool)
    bug_list = [bug_initial, bug_secondary, bug_tertiary, bug_quaternary, nobug_5ary, bug_5ary]
    for bugfunc in bug_list:
        bugfuncname = bugfunc.__name__
        try:
            bugfunc()
            print(f"{bugfuncname} raised no error.", file=stderr)
        except IndexError as e:
            print(f"{bugfuncname} raised the IndexError {e}", file=stderr)
    # Populate the namespace with the resulting variables of `bug_5ary`
    img, ndim, offsets, coords, hull, vertices, hull_perim_r, hull_perim_c, mask = bug_5ary(rets=True)
