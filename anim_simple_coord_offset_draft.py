from sys import stderr
import numpy as np
from skimage.measure import regionprops
from skimage.morphology.convex_hull import _offsets_diamond
from sample_data import SAMPLE
from common_subroutines import common_subroutine_1, common_subroutine_2
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

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

# Generate frames
edit_img = np.copy(img).astype(int) * 255
seen = []
frames = []

x_min, y_min = 0, 0
x_max, y_max = np.subtract(img.shape, 1)
not_l = pre_coords[:,0] != x_min
not_r = pre_coords[:,0] != x_max
not_t = pre_coords[:,1] != y_min
not_b = pre_coords[:,1] != y_max

excludes = [not_l, not_r, not_t, not_b]
# frames.append(edit_img.copy()) # initial
for exclude in excludes:
    exc_coords = pre_coords[exclude]
    exc_r, exc_c = exc_coords[:,0], exc_coords[:,1]
    img_exc = edit_img[exc_r, exc_c]
    mid_indx = np.argwhere(np.logical_and(0 < img_exc, img_exc < 255)).reshape(-1)
    max_indx = np.argwhere(img_exc == 255).reshape(-1)
    mid_r, mid_c = exc_r[mid_indx], exc_c[mid_indx]
    max_r, max_c = exc_r[max_indx], exc_c[max_indx]
    edit_img[mid_r, mid_c] += 40
    if frames and not np.array_equal(edit_img, frames[-1]):
        frames.append(edit_img.copy()) # update
    elif not frames:
        frames.append(edit_img.copy()) # update
    edit_img[max_r, max_c] = 40
    if frames and not np.array_equal(edit_img, frames[-1]):
        frames.append(edit_img.copy()) # update
    include = np.invert(exclude)
    include_coords = pre_coords[np.argwhere(include).reshape(-1)]
    seen.append(include_coords)

fig = plt.figure()
shape = edit_img.shape
ax = plt.axes(xlim=(0, shape[1]-1), ylim=(0, shape[0]-1))
a = edit_img
im = plt.imshow(a, interpolation="none", cmap=plt.get_cmap("magma"))

def init():
    im.set_data(a)
    return [im]

# animation function.  This is called sequentially
def update(frame):
    im.set_array(frame)
    return [im]

ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=100)
ani.save("edge_omit_fail_animation.gif", writer="imagemagick", fps=10)
plt.show()
