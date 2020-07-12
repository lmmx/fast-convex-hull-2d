import cProfile
import numpy as np
import tifffile
from PIL.Image import frombytes
from pathlib import Path

from skimage.morphology import convex_hull_image


#test_dir = 'C:/Users/husby036/Documents/Cprojects/test_s2s/testFiles/'
test_dir = Path("~/dev/convhull2d/cprof/").expanduser()


def saveImage(array, path):
    if array.dtype == np.bool:
        img = frombytes(mode='1', size=array.shape[::-1], data=np.packbits(array, axis=1))
        img.save(path)
    else:
        tifffile.imsave(path, array)
    print(f"'{path}' saved")


function = "convex_hull_image(image, offset_coordinates=True)"

chull_type = "latest"

image = tifffile.imread(test_dir / "image.tif")
print(f"{function} [{chull_type}]:\n")
cProfile.run(f"image_chull = {function}")
saveImage(image_chull, test_dir / f"chull_{chull_type}.tif")

debug_mask = np.zeros(image.shape, dtype=np.int8)
debug_mask[image_chull] = 1
debug_mask[image] += 2
saveImage(debug_mask, test_dir / f"chull_{chull_type}_inspection.tif")
