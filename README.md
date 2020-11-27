# fast-conv-hull-2d

:running: SciPy 2020 sprint on scikit-image for a faster 2D convex hull algorithm :running:

There was a pull request (PR) proposing a faster convex hull algorithm, but it gave some
errors. This is discussed [in the sprint notes](sprint_notes.md), which includes an expository
intro with some literature references.

Now that the source of the bug was discovered during the sprint, the task is now to
fix it.

(Work in progress)

- The last commit was "Cannot use cprofiler script without input tif"
- Note to self: see also `~/dev/skimage-pr-2928` and `/home/louis/dev/skimage-patch/skimage/morphology/convex_hull.py`
- Closed an open terminal with the following:

```
(skimage-dev) louis $ ~/dev/skimage $ python
Python 3.8.3 | packaged by conda-forge | (default, Jun  1 2020, 17:43:00)
[GCC 7.5.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>
>>> from skimage.draw import polygon_perimeter
>>> img = np.zeros((10,10), dtype=np.uint8)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'np' is not defined
>>> import numpy as np
>>> img = np.zeros((10,10), dtype=np.uint8)
>>> rr, cc = polygon_perimeter([5, -1, 5, 10], [-1, 5, 11, 5], shape=img.shape, clip=True)
>>> img[rr, cc] = 1
>>> img
array([[0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
       [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
       [0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
       [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=uint8)
```

Meanwhile, in another terminal

```
(skimage-dev) louis 🚶 ~/dev/skimage-patch $ python -i test_convex_area.py
bug_initial raised the IndexError index 10 is out of bounds for axis 0 with size 10
bug_secondary raised the IndexError index 10 is out of bounds for axis 0 with size 10
bug_tertiary raised the IndexError index 10 is out of bounds for axis 0 with size 10
bug_quaternary raised the IndexError index 10 is out of bounds for axis 0 with size 10
nobug_5ary raised no error.
bug_5ary raised the IndexError index 10 is out of bounds for axis 0 with size 10
>>> coords
>>> coords
array([[-0.5,  9. ],
       [ 0.5,  9. ],
       [ 0. ,  8.5],
...
       [ 9.5, 17. ],
       [ 9. , 16.5],
       [ 9. , 17.5]])
>>>
>>> coords.max(axis=0)
array([ 9.5, 17.5])
>>> print(coords.max(axis=0))
[ 9.5 17.5]
>>> print(coords.max(axis=0).tolist())
[9.5, 17.5]
```

I saved a vim file which was open and now there are committed diffs in the file,
see `test_convex_area` in `~/dev/skimage-patch`
