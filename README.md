# fast-conv-hull-2d

:running: SciPy 2020 sprint on scikit-image for a faster 2D convex hull algorithm :running:

## Intro: choosing an algorithm

One fast algorithm for computing the convex hull in 2D is known as "Graham's scan", published at Bell Labs

- R. L. Graham (1972) [An Efficient Algorithm for Determining the Convex Hull of a Finite Planar Set](http://www.math.ucsd.edu/~ronspubs/72_10_convex_hull.pdf)

...another algorithm was implemented by Graham's colleague, [Kenneth L. Clarkson](https://en.wikipedia.org/wiki/Kenneth_L._Clarkson),
available from his old Bell Labs website ([archived here](https://web.archive.org/web/19980715014112/http://cm.bell-labs.com/cm/cs/who/clarkson/2dch.c)),
- (“known for his research in computational geometry… co-editor-in-chief of the _Journal of Computational Geometry_”).
- The code was listed [on his profile](https://web.archive.org/web/20081024042432/http://cm.bell-labs.com/who/clarkson/)
  as “a short, complete planar convex hull code”.
- It is included here (along with the license header) as [`2dch.c`](2dch.c)

I'm not sure how to run this C code (`gcc` gives an error about integer arguments being of the wrong type), but it gives a general idea.

There was a pull request (PR) #2928 proposing a faster convex hull algorithm, but it gave some
errors. I'll write more about those below.

## Reviewing scikit-image PR

- [#2928: "Faster convex_hull_image polygon drawing for 2D images`"](https://github.com/scikit-image/scikit-image/pull/2928)

This thread begins with a proposal to change the convex hull calculation in scikit-image (henceforth "skimage").

It isn't accepted as it introduces errors, which are caught by the Travis CI testing pipeline:

First, at [line 4433](https://travis-ci.org/github/scikit-image/scikit-image/jobs/319439828#L4433):

```py
            hull_perim_r, hull_perim_c = polygon_perimeter(
                vertices[:, 0], vertices[:, 1]
            )

            mask = np.zeros(image.shape, dtype=np.bool)

>           mask[hull_perim_r, hull_perim_c] = True
```
⇣
```
IndexError: index 10 is out of bounds for axis 0 with size 10
```

i.e. here we have a
`polygon_perimeter` ([source](https://github.com/scikit-image/scikit-image/blob/d325ef60c2ae7404c664cfa301e0865c9ae15c96/skimage/draw/draw.py#L229),
[docs](https://scikit-image.org/docs/stable/api/skimage.draw.html#skimage.draw.polygon_perimeter)) function call, which returns
into 2 variables `hull_perim_r` and `hull_perim_c`

> #### Returns
>
> **rr**, **cc**: `ndarray` of `int`
>
> Pixel coordinates of polygon. May be used to directly index into an array, e.g.
> `img[rr, cc] = 1`.
>
> #### Examples
>
> ```py
>>> from skimage.draw import polygon_perimeter
>>> img = np.zeros((10, 10), dtype=np.uint8)
>>> rr, cc = polygon_perimeter([5, -1, 5, 10],
...                            [-1, 5, 11, 5],
...                            shape=img.shape, clip=True)
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
