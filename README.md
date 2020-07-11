# fast-conv-hull-2d

:running: SciPy 2020 sprint on scikit-image for a faster 2D convex hull algorithm :running:

There was a pull request (PR) proposing a faster convex hull algorithm, but it gave some
errors. This is discussed [below](#reviewing-scikit-image-pr), after a quick expository intro.

## Intro: choosing what to "sprint" on

Some 'requested features' are listed
[here](https://github.com/scikit-image/scikit-image/wiki/Requested-features) (and anyone can
[contribute](https://scikit-image.org/docs/stable/contribute.html), but this weekend July 11th-12th
2020 it's SciPy 2020's [“sprints”](https://www.scipy2020.scipy.org/sprints)).

One possible algorithm to implement listed was:

> - Fast 2D convex hull (consider using CellProfiler version).
>   - [Algorithm overview](https://web.archive.org/web/20100306010010/http://www.tcs.fudan.edu.cn/rudolf/Courses/Algorithms/Alg_cs_07w/Webprojects/Zhaobo_hull/index.html#section26).
>   - [One free implementation](https://web.archive.org/web/19980715014112/http://cm.bell-labs.com/cm/cs/who/clarkson/2dch.c).
>     - (Compare against current implementation.)

This stood out to me as something worth improving as convex hulls are one of the basics of
convex optimisation, and also I knew of the [_CellProfiler_]() project (belonging to Anne Carpenter
at the Broad Institute).

The "CellProfiler version" was a reference to
https://github.com/CellProfiler/centrosome/blob/master/centrosome/_convex_hull.pyx

In fact, this feature request had been in the Wiki since this page's creation
[all the way back in 2012](https://github.com/scikit-image/scikit-image/wiki/Requested-features/7e47b11e3bdb5245b9c6676e776c6745fc265124)!

- This initial version noted that there was ongoing work to ["merge code provided by CellProfiler
  team"](https://github.com/scikit-image/scikit-image/wiki/Requested-features/7e47b11e3bdb5245b9c6676e776c6745fc265124#merge-code-provided-by-cellprofiler-team)
- The code provided was 2 files of what is now a GitHub repo but was then a Broad Institute SVN trunk:
  - [`cellprofiler/cpmath/cpmorphology.py`](https://github.com/CellProfiler/centrosome/blob/master/centrosome/cpmorphology.py)
    and [`cellprofiler/cpmath/filter.py`](https://github.com/CellProfiler/centrosome/blob/master/centrosome/filter.py) which are still
    in the repo today!

Having chosen this, I did a little literature review (summarised in the next section) and then began
to review the problems with the pull request (click [here](#reviewing-scikit-image-pr) to jump to that).

## Intro: choosing an algorithm/literature review

One fast algorithm for computing the convex hull in 2D is known as "Graham's scan", published at Bell Labs

- R. L. Graham (1972) [An Efficient Algorithm for Determining the Convex Hull of a Finite Planar Set](http://www.math.ucsd.edu/~ronspubs/72_10_convex_hull.pdf)

...another algorithm was implemented by Graham's colleague, [Kenneth L. Clarkson](https://en.wikipedia.org/wiki/Kenneth_L._Clarkson),
available from his old Bell Labs website ([archived here](https://web.archive.org/web/19980715014112/http://cm.bell-labs.com/cm/cs/who/clarkson/2dch.c)),
- (“known for his research in computational geometry… co-editor-in-chief of the _Journal of Computational Geometry_”).
- The code was listed [on his profile](https://web.archive.org/web/20081024042432/http://cm.bell-labs.com/who/clarkson/)
  as “a short, complete planar convex hull code”, and he notes he is "particularly" interested in
  "algorithms that have provable properties, but are relatively simple" (indeed he coauthored a
  paper with Shor of [Shor's algorithm](https://en.wikipedia.org/wiki/Shor%27s_algorithm) fame)
- It is included here (along with the license header) as [`2dch.c`](2dch.c)

I'm not sure how to run this C code (`gcc` gives an error about integer arguments being of the wrong type), but it gives a general idea.

Some books which cover this subject:

- :book: Bærentzen (2012) Guide to Computational Geometry Processing
  - Chapter 13: Convex Hulls; section 13.3: Convex Hull Algorithms in 2D
- :book: de Berg (2008) Computational Geometry: Algorithms and Applications
  - Chapter 1, p.13
- :book: Devados (2011) Discrete and Computational Geometry
  - Chapter 2: Convex Hulls
- :book: O'Rourke (1998) Computational Geometry in C
  - Chapter 3: Convex Hulls in Two Dimensions

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
> >>> from skimage.draw import polygon_perimeter
> >>> img = np.zeros((10, 10), dtype=np.uint8)
> >>> rr, cc = polygon_perimeter([5, -1, 5, 10],
> ...                            [-1, 5, 11, 5],
> ...                            shape=img.shape, clip=True)
> >>> img[rr, cc] = 1
> >>> img
> array([[0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
>        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
>        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
>        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
>        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
>        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
>        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
>        [0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
>        [0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
>        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=uint8)
> ```
