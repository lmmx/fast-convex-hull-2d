from sys import stderr
import numpy as np
from skimage.measure import regionprops
from skimage.morphology.convex_hull import convex_hull_image
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

def bug_5ary(img=None):
    if img is None:
        img = SAMPLE.astype(np.bool)
    coords = possible_hull(np.ascontiguousarray(img, dtype=np.uint8))
    hull = ConvexHull(coords)
    vertices = hull.points[hull.vertices]
    hull_perim_r, hull_perim_c = polygon_perimeter(vertices[:, 0], vertices[:, 1])
    mask = np.zeros(img.shape, dtype=np.bool)
    # This is the line that reportedly raises IndexError but it doesn't here ? ? ?
    mask[hull_perim_r, hull_perim_c] = True
    return

if __name__ == "__main__":
    rp = regionprops(SAMPLE)[0]
    img = SAMPLE.astype(np.bool)
    bug_list = [bug_initial, bug_secondary, bug_tertiary, bug_quaternary, bug_5ary]
    for bugfunc in bug_list:
        bugfuncname = bugfunc.__name__
        try:
            bugfunc()
            print(f"{bugfuncname} raised no error.", file=stderr)
        except IndexError as e:
            print(f"{bugfuncname} raised the IndexError {e}", file=stderr)
