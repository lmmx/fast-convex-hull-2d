# fast-conv-hull-2d

:running: SciPy 2020 sprint on scikit-image for a faster 2D convex hull algorithm :running:

There was a pull request (PR) proposing a faster convex hull algorithm, but it gave some
errors. This is discussed [below](#reviewing-scikit-image-pr), after a quick expository intro.

## Intro: choosing what to "sprint" on

Some 'requested features' are listed
[here](https://github.com/scikit-image/scikit-image/wiki/Requested-features) (and anyone can
[contribute](https://scikit-image.org/docs/stable/contribute.html), but this weekend July 11th-12th
2020 it's SciPy 2020's [“sprints”](https://www.scipy2020.scipy.org/sprints)).

- Some issues were also labelled '[sprint](https://github.com/scikit-image/scikit-image/issues?q=is%3Aissue+is%3Aopen+label%3Asprint)'

One possible algorithm to implement listed was:

> - Fast 2D convex hull (consider using CellProfiler version).
>   - [Algorithm overview](https://web.archive.org/web/20100306010010/http://www.tcs.fudan.edu.cn/rudolf/Courses/Algorithms/Alg_cs_07w/Webprojects/Zhaobo_hull/index.html#section26).
>   - [One free implementation](https://web.archive.org/web/19980715014112/http://cm.bell-labs.com/cm/cs/who/clarkson/2dch.c).
>     - (Compare against current implementation.)

This stood out to me as something worth improving as convex hulls are one of the basics of
convex optimisation, and also I knew of the [_CellProfiler_]() project (belonging to Anne Carpenter
at the Broad Institute).

The "CellProfiler version" seemed to be a reference to the Cython file: [`_convex_hull.pyx`](https://github.com/CellProfiler/centrosome/blob/master/centrosome/_convex_hull.pyx)...

...In fact, this feature request had been in the Wiki since this page's creation
[all the way back in 2012](https://github.com/scikit-image/scikit-image/wiki/Requested-features/7e47b11e3bdb5245b9c6676e776c6745fc265124)!

This initial version noted that there was ongoing work to ["merge code provided by CellProfiler
team"](https://github.com/scikit-image/scikit-image/wiki/Requested-features/7e47b11e3bdb5245b9c6676e776c6745fc265124#merge-code-provided-by-cellprofiler-team)

The code provided to `scikit-image` under a BSD licence was 2 files of what is now a GitHub repo but was then a Broad Institute SVN trunk:

- [`cellprofiler/cpmath/cpmorphology.py`](https://github.com/CellProfiler/centrosome/blob/master/centrosome/cpmorphology.py)
- [`cellprofiler/cpmath/filter.py`](https://github.com/CellProfiler/centrosome/blob/master/centrosome/filter.py)

...which are still in the repo to this day!

Having chosen this, I did a little literature review (summarised in the next section) and then began
to review the problems with the pull request (click [here](#reviewing-scikit-image-pr) to jump to that).

## Intro: choosing an algorithm/literature review

One fast algorithm for computing the convex hull in 2D is known as "Graham's scan", published at Bell Labs

- R. L. Graham (1972) [An Efficient Algorithm for Determining the Convex Hull of a Finite Planar Set](http://www.math.ucsd.edu/~ronspubs/72_10_convex_hull.pdf)

> Given a finite set S = {s₁, ..., sₙ} in the plane…
>
> Step 1: Find a point P in the plane which is in the interior of CH(S) [the convex hull of S]…
>
> Step 2: Express each sᵢ ∈ S in polar coordinates with origin P and 𝜃 = 0 in the direction of an
> arbitrary fixed half-line L from P…
>
> Step 3: Order the elements 𝞀ₖ exp(i𝜃ₖ) of S in terms of increasing 𝜃ₖ…
>
> We now have S in the form S = {r₁exp(iφ₁), ..., rₙexp(iφₙ)} with 0 ≤ φ₁ ≤ ... ≤ φₙ < 2π and rᵢ ≥ 0…
>
> Step 4: If φᵢ = φᵢ₊₁ then we may delete the point with the smaller amplitude since it clearly
> cannot be an extreme point of CH(S). Also any point with rᵢ = 0 can be deleted.
>
> ...By relabelling the remaining points, we can set Sʹ = {r₁exp(iφ₁), ..., rₙˊexp(iφₙˊ)} where nʹ ≤ n.
>
> Step 5: Start with three consecutive points in Sʹ... There are two possibilities:
> (i) α+β ≥ π. Then we delete the point rₖ₊₁...
> (ii) α+β ≤ π. Return to the beginning of Step 5 [replacing some points]...
>
> By noting that each application of step 5 _either_ reduces the number of possible points of CH(S)
> by one _or_ increases the current total number of points of Sʹ considered by one, **an easy
> induction argument shows that with less than 2nʹ iterations of step 5, we must be left with
> exactly the subset of S of all extreme points of CH(S). This completes the algorithm.**
>
> The reader may find it instructive to consider a small example of ten points or so. Computer
> implementation of this algorithm makes it quite feasible to consider examples with n = 50 000.

Some examples of this exist on GitHub ([here](https://github.com/search?l=Python&q=graham%27s+scan&type=Repositories)), so one option
could be to implement this as a fast 2D convex hull calculator.

- e.g. [this one](https://github.com/ejydavis/pyGrahamScan/blob/master/grahamscan.py) is 60 lines.

Another algorithm was implemented by Graham's colleague, [Kenneth L. Clarkson](https://en.wikipedia.org/wiki/Kenneth_L._Clarkson),
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

To review the PR including its file attachments (which are not version controlled!) on GitHub, I
clicked 'quote reply' and the little `⋯` icon in the top-right of the PR description, then copied
the text into [`2928.md`](2928.md) and trimmed off the prefixing `> ` (the quote block) to get the
original description as a local markdown file.

This file can be converted to HTML with:

```sh
(echo "<html><body>"; pandoc 2928.md; echo "</html></body>") > 2928.html
```

(I thought I could then `wget` all the links but seemingly not)

Better yet we can download the page and all of its attachments with `wget`:

```sh
mkdir 2928_files
cd 2928_files
wget -nd -rk --include-directories="scikit-image/scikit-image/files/" -l 1 https://github.com/scikit-image/scikit-image/pull/2928
rm robots.txt
mv 2928 2928.html
```

The files with names ending in `_cprof.txt` are results from [cProfile](https://docs.python.org/3.8/library/profile.html),

> [`cProfile`](https://docs.python.org/3.8/library/profile.html#module-cProfile) and [`profile`](https://docs.python.org/3.8/library/profile.html#module-profile) provide _deterministic profiling_ of Python programs. A _profile_ is a set of
> statistics that describes how often and for how long various parts of the program executed. These
> statistics can be formatted into reports via the
> [`pstats`](https://docs.python.org/3.8/library/profile.html#module-pstats) module.

I manually excised the script used to generate these from the backtick block in `2928.md` into
[`2928_files/pr_cprofiler.py`](2928_files/pr_cprofiler.py) (which uses `cProfile`),
and changed its hardcoded `test_dir` to one on my machine (`~/dev/convhull2d/cprof/`) and rewrote
the code to use f-strings and `pathlib.Path`.

It turns out this can't be run, as the input `tif` file being used in testing wasn't provided as a
file attachment (`image.tif`). Let's press on anyway.


### What was in the PR?

- [#2928: "Faster convex_hull_image polygon drawing for 2D images"](https://github.com/scikit-image/scikit-image/pull/2928)

This thread begins with a proposal to change the convex hull calculation in scikit-image (henceforth "skimage").

Before we look at why it wasn't accepted, first off let's note that the PR consists of
[3 commits](https://github.com/scikit-image/scikit-image/pull/2928/commits), the first of which
contains the actual code, the 2nd is purely formatting, and the 3rd is a branch merge ("cleanly"
with no code modifications involved).

- The first commit is ["Faster convex polygon drawing for 2D images"](https://github.com/scikit-image/scikit-image/pull/2928/commits/d40f773f8f02bb2be766a46c8202cd200ca3c35f) (20th December 2017).
- The commit message was:

> Replaced `grid_points_in_poly` with calls to `skimage.draw.polygon_perimeter` and
> `scipy.ndimage.morphology.binary_fill_holes` in convex polygon drawing step for a 2D image.
>
> For large 2D images (~10,000 x ~10,000 pixels), this substitution can result in a
> function-call-to-return speedup of more than 5x (from 23.0 sec to 4.4 sec for a particular image
> with about 150 convex hull edges) while producing a convex hull image that is nearly identical to
> the image created by the current drawing routine.  In following comments, I will compare the two
> results of these two routines.

This tells us that it gave a speedup, but how? That's in the code comments itself: the line

> \# If 2D, use fast Cython function to locate convex hull pixels

was removed in favour of:

> \# If 2D, locate hull perimeter pixels and use fast SciPy function to fill it in

i.e. we're still using that fast Cython function (the aforementioned
[`_convex_hull.pyx`](https://github.com/CellProfiler/centrosome/blob/master/centrosome/_convex_hull.pyx) that came from
CellProfiler and was incorporated into scikit-image), but furthermore we're using a fast SciPy
function:

```py
from scipy.ndimage.morphology import binary_fill_holes
```

Whose docs are
[here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_fill_holes.html):

> `scipy.ndimage.binary_fill_holes(input, structure=None, output=None, origin=0)`
> 
> Fill the holes in binary objects.

A potentially useful parameter to note (that isn't used in the PR) is `output`:

> `output` ndarray, optional
>
> Array of the same shape as input, into which the output is placed. By default, a new array is created.

i.e. if we pass an `output` argument we can assign to that rather than creating a new array, and
maybe make a more lightweight function (but that's for consideration once it works).

The example in the docs is:

```py
>>> a
array([[0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 0, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 0, 0, 0, 0]])
>>> ndimage.binary_fill_holes(a).astype(int)
array([[0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 0, 0, 0, 0]])
```

The docs note:

> The algorithm used in this function consists in invading the complementary _[sic]_ of the shapes in input
> from the outer boundary of the image, using binary dilations. Holes are not connected to the
> boundary and are therefore not invaded. The result is the complementary subset of the invaded
> region.

'Complementary' should be 'complement', i.e. it inverts the binary image so 'holes' become
positive and are not "connected to the boundary" (only positive pixels can be connected)
so do not get "invaded" by the (presumably iterative)
[binary dilation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_dilation.html#scipy.ndimage.binary_dilation).

The code change itself is just the introduction of 3 new lines.

The assignment of `mask`:

```py
mask = grid_points_in_poly(image.shape, vertices)
```

is swapped out for:

```py
hull_perim_r, hull_perim_c = polygon_perimeter(vertices[:, 0], vertices[:, 1])
mask = np.zeros(image.shape, dtype=np.bool)
mask[hull_perim_r, hull_perim_c] = True
mask = binary_fill_holes(mask)
```

- The first of those new lines sets 2 variables for the row and column of the CH perimeter,
  [the docs](https://scikit-image.org/docs/stable/api/skimage.draw.html#skimage.draw.polygon_perimeter)
  for which note "May be used to directly index into an array"

- The mask is initialised as all `False`, with the same shape as the image whose CH is to be found

- The mask is set to `True` (this is the aforementioned direct indexing)

- The holes in the mask are filled with the SciPy function

### What happened to the PR after submission?

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

So you pass in some polygon points and these are interpolated on a discrete grid whose
dimensions match those of some `img` (an all-zero array) to give the binary
image of the outline of a polygon when directly indexed into.

The `shape` parameter is optional, and defaults to `None`.

> If None, the full extents of the polygon is used. Must be at least length 2.

We can see that in the code above, it's not provided so will be `None`.

The error is thrown from the 0'th axis, i.e. the row, i.e. `hull_perim_r`,
and furthermore the context shows that it does so in
[`test_regionprops.py`](https://github.com/scikit-image/scikit-image/blob/master/skimage/measure/tests/test_regionprops.py),
the test for [`skimage.measure.regionprops`](https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops)

```sh
find ./ -iname test_regionprops.py 2> /dev/null
```
⇣
```STDOUT
./skimage/measure/tests/test_regionprops.py
```

This function `regionprops` computes 2 things reliant on 2D convex hull

> **convex_area** : int
>
> Number of pixels of convex hull image, which is the smallest convex polygon that encloses the
> region.
> 
> **convex_image(H, J)** : ndarray
>
> Binary convex hull image which has the same size as bounding box.

Specifically, it errors at "line 121" (currently actually [line
152](https://github.com/scikit-image/scikit-image/blob/master/skimage/measure/tests/test_regionprops.py#L152)),

```py
def test_convex_area():
    area = regionprops(SAMPLE)[0].convex_area
```

...where `SAMPLE` is a hardcoded array (line
[16](https://github.com/scikit-image/scikit-image/blob/master/skimage/measure/tests/test_regionprops.py#L16-L27) of the same file)

```py
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
```

This is good then, we can immediately make a script [`test_convex_area.py`](test_convex_area.py) to reproduce the error.

I couldn't see this but maintainers can provide the command to run to get up to date with a given
PR, which in this case was:

```sh
git checkout -b ehusby-patch-1 master
git pull https://github.com/ehusby/scikit-image.git patch-1
```

```sh
python test_convex_area.py
```
⇣
```STDOUT
Traceback (most recent call last):
  File "test_convex_area.py", line 22, in <module>
    test_convex_area()
  File "test_convex_area.py", line 18, in test_convex_area
    area = regionprops(SAMPLE)[0].convex_area
  File "/home/louis/dev/skimage-patch/skimage/measure/_regionprops.py", line 190, in wrapper
    cache[prop] = f(obj)
  File "/home/louis/dev/skimage-patch/skimage/measure/_regionprops.py", line 297, in convex_area
    return np.sum(self.convex_image)
  File "/home/louis/dev/skimage-patch/skimage/measure/_regionprops.py", line 190, in wrapper
    cache[prop] = f(obj)
  File "/home/louis/dev/skimage-patch/skimage/measure/_regionprops.py", line 303, in convex_image
    return convex_hull_image(self.image)
  File "/home/louis/dev/skimage-patch/skimage/morphology/convex_hull.py", line 90, in
convex_hull_image
    mask[hull_perim_r, hull_perim_c] = True
IndexError: index 10 is out of bounds for axis 0 with size 10
```

:tada: The error is now minimally reproducible! We can take a look at the source code for
`regionprops` to understand what `regionprops(SAMPLE)[0].convex_area` is doing

The basic route to obtaining the function `regionprops` goes:

- `from skimage.measure import regionprops` takes the name `regionprops` from the namespace of
  the `skimage.measure` submodule which exports an `__all__` variable in its `__init__.py` file
  with "regionprops" (line 19)
  - The `__init__` file is able to export this name in an `__all__` variable because on line 5
    it imported it `from ._regionprops` (i.e. the internal file `skimage.measure._regionprops`)
- The file `_regionprops.py` also declares an `__all__` variable, meaning it limits what it exports
  to the namespace, on line 17 (in which again `'regionprops'` is declared as a string in a list.
  This time it is not exporting something imported, but rather a funcdef defined on line 806 which
  returns `regions`, a "list of `RegionProperties`" each of which "describes one labelled region"
  (i.e. when we access `regionprops(SAMPLE)[0]` we access the first `RegionProperties` object in
  the singleton list).

So in other words, if we first assign:

```py
rp = regionprops(SAMPLE)[0]
```

then `rp` is a `RegionProperties` object... but what is a `RegionProperties` object?

Well it's created on line 1084, as

```py
props = RegionProperties(sl, label, label_image, intensity_image,
                         cache, extra_properties=extra_properties)
```

and the class constructor being called there is not imported, but was defined on line 207

```py
class RegionProperties:
    def __init__(self, slice, label, label_image, intensity_image,
                 cache_active, *, extra_properties=None):
        ...
```

The `def` line for the funcdef `regionprops` (line 806) is:

```py
def regionprops(label_image, intensity_image=None, cache=True,
                coordinates=None, *, extra_properties=None)
```

So since we call `regionprops` with only one argument (`SAMPLE`), this becomes the argument
`label_image` which is passed to `RegionProperties` as the `slice` argument, then the rest of the parameters
of the `regionprops` funcdef take their default arguments:

- `intensity_image=None` --> passed directly to the class constructor `RegionProperties`
- `cache=True`
- `coordinates=None`
- `extra_properties=None` --> passed directly to the class constructor `RegionProperties`

In other words the thing to pay attention to is the 3rd argument [ignoring `self`] of the
`RegionProperties` constructor, `label_image`, which we are setting as our array `SAMPLE`.

...and now finally we get to the root of the attribute: like the class name suggests, the
attributes defined on `RegionProperties` are actually properties (i.e. their method definitions
carry the `@property` decorator (from the `functools` library), and `convex_area` is the one
on lines 294-297:

```py
@property
@_cached
def convex_area(self):
   return np.sum(self.convex_image)
```

So at a guess, surely `sum` wouldn't be erroring, so it must be that when `self.convex_image` is
accessed then this too is a property...

Indeed, accessing `rp.convex_image` throws the same error, so we can switch the function
`bug_initial` which accesses `rp.convex_area` for another function `bug_initial_cause` which
reproduces the same bug by instead accessing `rp.convex_image`.

So what happens when the `convex_image` attribute is accessed? Presumably only a property which runs
a function upon access would produce an error upon access.

Of course, `convex_image` is the next defined (property-decorated) method, on line 299-303:

```py
@property
@_cached
def convex_image(self):
    from ..morphology.convex_hull import convex_hull_image
    return convex_hull_image(self.image)
```

So this is going 'up a level' from `skimage.measure` and into the `skimage.morphology` submodule,
and then the
[`convex_hull.py`](https://github.com/scikit-image/scikit-image/blob/master/skimage/morphology/convex_hull.py) submodule file,
and specifically to [line 21](https://github.com/scikit-image/scikit-image/blob/master/skimage/morphology/convex_hull.py#L21) in that file,
where the function `convex_hull_image` is defined:

```py
def convex_hull_image(image, offset_coordinates=True, tolerance=1e-10):
    ...
    return mask
```

So again, we can simplify the initial bug, and for the first time we can find something to access on
the `rp` variable which doesn't raise an error: `rp.image` is the argument to `convex_hull_image`,
a `np.ndarray` of `dtype('bool')`, i.e. it's a binary mask, and in fact it's just the boolean typed
equivalent of the `SAMPLE` array (of integer dtype):

```py
>>> np.array_equal(SAMPLE, rp.image)
True
>>> rp.image.shape
(10, 18)
```

It's a 10 row, 18 column array just like `SAMPLE`.

So what part of `convex_hull_image` is mis-indexing our array?

- Specifically, something is trying to access the 11th row of a 10 row array, leading to
  this `IndexError: index 10 is out of bounds for axis 0 with size 10` message

The good news is that we've managed to simplify the bug 4 times now, it can be stated as
given in the function `bug_quaternary`:

```py
img = SAMPLE.astype(np.bool)
conv_img = convex_hull_image(img)
```

which will throw the exact same `IndexError` as the initial 'draft'/appearance of the bug,
as given in the function `bug_initial` (as `regionprops(SAMPLE)[0].convex_area`).
