from pythonrc import *
from simple_test_convex_area import *
from matplotlib import pyplot as plt

def round(decimal):
    "Round away from zero"
    sgn = np.sign(decimal)
    r = np.ceil(np.abs(decimal))
    return (sgn * r).astype(int)

img = img.astype(int) * 255
for (row,col) in coords:
    rrow, rcol = map(round, [row, col])
    print(f"{rrow}, {rcol}")
plt.imshow(img)
