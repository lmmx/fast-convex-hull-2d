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
convex optimisation, and also I knew of the
[_CellProfiler_](https://en.wikipedia.org/wiki/CellProfiler) project (belonging to Anne Carpenter
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
- :book: Klette & Rosenfeld (2004) [_Digital Geometry: Geometric Methods for Digital Picture Analysis_](https://books.google.co.uk/books?id=4_iEl6cquGYC&pg=PA432&lpg=PA432#v=onepage&q&f=false)
  - Chapter 13: Hulls and diagrams, especially 13.1.2 Convex hull computation in the 2D grid
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

- This can be visualised as an image with matplotlib `pyplot.imshow`, which shows it's generally
  kind of like an equilateral triangle with a concave base, like the Star Trek logo
- Also note that there are no empty rows or columns at the edges of this array (so the convex hull
  will be an array of the same size)

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

The error arises at line 90 in the patch (i.e. the merged development version on my machine), and
[line 86 in ehusby's `patch-1` branch](https://github.com/ehusby/scikit-image/blob/patch-1/skimage/morphology/convex_hull.py#L86)
(which is what Travis CI runs its tests on therefore the line number it reports).

- ehusby's fork is now "2537 commits behind scikit-image:master" at the time of writing, hence the
  difference in line numbering.
- From now on I will list line numbers as "current patch (ehusby's patch)"
  - e.g. in this case, the error arises at line 90 (line 86).
  - I can only give a URL link to ehusby's patch (GitHub won't display a new patch as far as I know)
  - If I don't give a line number in brackets then I didn't bother looking it up, and if I write
    "(=)" then the line numbers match.

So anyway, what is at line 90 (line 86) that causes this error?

```py
mask[hull_perim_r, hull_perim_c] = True
```

There are 3 variables involved, and the first 2 are where the error arises (the row number
`hull_perim_r` is apparently `10`, and this is the 11th [0-based] index into an array of 10 rows,
hence the `IndexError`.

We've been over this before, when we looked at the changes introduced.

The very first line of these changes was

```py
hull_perim_r, hull_perim_c = polygon_perimeter(vertices[:, 0], vertices[:, 1])
```

and the function generating these variables was `polygon_perimeter` using `vertices`,
so we need to look at what `vertices` are (as `polygon_perimeter` is "just" a function, and so we
need to get to `vertices` from our `img` which is just a bool-type `SAMPLE`.

How do we get the `vertices`? Well it turns out it's 2 lines, lines 80-81 of my patched repo,
and it relates to the `ConvexHull` class:

```py
hull = ConvexHull(coords)
vertices = hull.points[hull.vertices]
```

This should correspond to the "equilateral triangle with a concave base"-like image we expected
earlier (I said it looked a bit like the Star Trek logo). Importantly, there were no empty rows
or columns around the perimeter, so the convex hull will be the same size as the input array
(i.e. it won't be cropped during this process).

The `ConvexHull` class came from `scipy.spatial`, imported at line 4 (=).

This time, `ConvexHull` is not visibly defined as a string literal in `scipy.spatial.__init__`,
but can be found at line 2267 of `qhull.pyx`, which is imported into `scipy.spatial.__init__` on
line 98:

```py
from .qhull import *
```

and then exported into the `__all__` variable on line 104 (via the `dir()` namespace):

```py
__all__ = [s for s in dir() if not s.startswith("_")]
```

The `.qhull` wildcard import refers to `qhull.pyx` (but not `qhull.pxd`, I think, these are
"definition files", the `.pyx` are "implementation files" - see [Cython ⠶ Language
basics](https://cython.readthedocs.io/en/latest/src/userguide/language_basics.html#cython-file-types)),
in which there is an `__all__` defined as a list literal containing a string literal
`'ConvexHull'`, hence it will be exported through the wildcard import into `scipy.spatial`.

The class itself is defined in `qhull.pyx` at line 2267, and here's where things start to look...
tricky.

The class definition of `ConvexHull` takes as constructor base class a class called `_QhullUser`.
The `__init__` method of the `ConvexHull` class has the method `def` line:

```py
def __init__(self, points, incremental=False, qhull_options=None)
```

So forgetting about everything else for a moment, how is this class constructor called?

It was called as `hull = ConvexHull(coords)` on line 80 of `skimage/morphology/convex_hull.py`,
the first of 2 lines which gave us the `vertices` which then fed into `polygon_perimeter` to produce
the `hull_perim_r` that caused the `IndexError`.

So according to the `__init__` method `def` line of `scipy.spatial.ConvexHull`, we should expect
this variable `coords` to be an instance of the class `_QhullUser`.

How is `coords` made? The `coords` variable gets assigned at multiple times (it seems to be the
iterative process you'd expect in an algorithm something like Graham's scan we encountered earlier).

Let's look for the first assignment: this is on line 60 (again, in `skimage/morphology/convex_hull.py`),
and it follows a check for `if ndim == 2` (indeed we are looking at the 2D/planar CH algorithm):

```py
coords = possible_hull(np.ascontiguousarray(image, dtype=np.uint8))
```

Here `coords` is assigned as the return value of the function `possible_hull`, and what I'd guess is
our trusty `img` variable (which is just a bool-typed `SAMPLE`).

This time, `possible_hull` didn't come from `scipy.spatial`, it was imported at line 7 of
`skimage/morphology/convex_hull.py`:

```py
from ._convex_hull import possible_hull
```

Again, this was a Cython implementation file, `_convex_hull.pyx`, and the function in question
(`possible_hull`) was defined on line 11, the `def` line of which is:

```py
def possible_hull(cnp.uint8_t[:, ::1] img):
```

...in which `cnp` comes from the import on line 7

```py
cimport numpy as cnp
```

In other words, `cnp` is Cython numpy.

The docstring for `possible_hull` adds that the input parameter `img` should be an "ndarray of
bool". [This](https://stackoverflow.com/a/46416257/2668831) StackOverflow answer gives some more
info:

> #### cnp.int_t
>
> It's the type-identifier version for `np.int_`. That means you can't use it as dtype argument. But you
> can use it as type for `cdef` declarations:
>
> ```
> cimport numpy as cnp
> import numpy as np
> 
> cdef cnp.int_t[:] arr = np.array([1,2,3], dtype=np.int_)
>      |---TYPE---|                         |---DTYPE---|
> ```
>
> This example (hopefully) shows that the type-identifier with the trailing `_t` actually represents
> the type of an array using the dtype without the trailing `t`. You can't interchange them in Cython
> code!
>
> ...
>
> ```
> NumPy dtype          Numpy Cython type         C Cython type identifier
> 
> np.bool_             None                      None
> np.int_              cnp.int_t                 long
> ```
>
> Actually there are Cython types for `np.bool_`: `cnp.npy_bool` and `bint` but both they can't be used
> for NumPy arrays currently. For scalars `cnp.npy_bool` will just be an unsigned integer while `bint`
> will be a boolean. Not sure what's going on there...

This is what we have here: our `img` (the bool-dtype version of the `SAMPLE` numpy array) is
represented by Cython `int_t` type, and additionally it's unsigned and 8 bit (hence `uint8_t`).

This then is the Cython data type being consumed to produce the variable `coords`, which I'll show
again:

```py
coords = possible_hull(np.ascontiguousarray(image, dtype=np.uint8))
```

As we can see, the `np.uint8` type will become `cnp.uint8_t` without anything further being done.
Neat!

The numpy function
[`ascontiguousarray`](https://numpy.org/doc/stable/reference/generated/numpy.ascontiguousarray.html)
seems to serve a special purpose for Cython-interacting numpy code:

> Return a contiguous array (ndim >= 1) in memory (C order).

In the example, it shows that apparently there is a 'flag' on an array called `"C_CONTIGUOUS"` which
will be reliably `True` for arrays created through `ascontiguousarray`, i.e. that's how you check
this property of being "a contiguous array in memory (C order)".

Interestingly, if we check `SAMPLE.flags["C_CONTIGUOUS"]` it returns `True`, so perhaps it's more of
a just-in-case type of function, for certain edge cases...

Interesting, but what is the outcome of all of this? What is `image` in the assignment to `coords`
for a start?

We met this earlier as the `def` line of `convex_hull_image`, at line 22 of
`skimage/morphology/convex_hull.py`, where it was being called on `self.image` to return the value
for the property `convex_image` in the first and second versions of the bug, but directly in `bug_secondary`:

```py
def convex_image(self):
    from ..morphology.convex_hull import convex_hull_image
    return convex_hull_image(self.image)
```

I.e. `self.image` is what we earlier defined into `bug_tertiary` as `img = rp.image`.

So we can now replace the assignment of `coords` with `rp.image` where `rp = regionprops(SAMPLE)[0]`,
or better yet, we can simplify this as we found in `bug_quaternary` that we could replace the entire
call to `regionprops` by just setting the `img` to `SAMPLE.astype(np.bool)`.

So in that case, we can rewrite the assignment of `coords` as:

```py
img = SAMPLE.astype(bool)
coords = possible_hull(np.ascontiguousarray(img, dtype=np.uint8))
```

and we can get the function `possible_hull` from `skimage/morphology/_convex_hull.py` like so:

```py
from skimage.morphology import _convex_hull
possible_hull = _convex_hull.possible_hull
```

(There's probably a better way to do that, but anyway...)

...and voila, it runs, we get `coords.shape` of `(53, 2)`

```py
>>> " ~ ".join([",".join(map(repr, c.tolist())) for c in coords])
'0,9 ~ 1,8 ~ 2,8 ~ 3,8 ~ 4,6 ~ 5,6 ~ 6,6 ~ 7,0 ~ 8,1 ~ 9,1 ~ 7,0 ~ 8,1 ~ 7,2 ~ 8,3 ~ 8,4 ~ 7,5 ~ 4,6
~ 4,7 ~ 1,8 ~ 0,9 ~ 1,10 ~ 0,11 ~ 0,12 ~ 5,13 ~ 5,14 ~ 5,15 ~ 5,16 ~ 8,17 ~ 0,12 ~ 1,11 ~ 2,10 ~
3,10 ~ 4,11 ~ 5,16 ~ 6,16 ~ 7,16 ~ 8,17 ~ 9,17 ~ 9,1 ~ 9,2 ~ 9,5 ~ 9,6 ~ 9,7 ~ 7,8 ~ 8,9 ~ 8,10 ~
6,11 ~ 7,12 ~ 7,13 ~ 8,14 ~ 8,15 ~ 9,16 ~ 9,17'
```

One thing to note about these coords is their range:

```py
>>> coords.ptp(axis=0)
array([ 9, 17])
```

i.e.

```py
>>> coords.min(axis=0).tolist(), coords.max(axis=0).tolist()
([0, 0], [9, 17])
```

Notice that the maximum row coordinate is 9, so we should never raise that `IndexError` that occurs
from accessing the nonexistent 10th row, so something must go wrong precisely after this assignment...

The next step was

```py
hull = ConvexHull(coords)
vertices = hull.points[hull.vertices]
```

where `ConvexHull` is a class from `scipy.spatial`. So far so good, we now have another array, `vertices`:

```py
array([[ 0., 12.],
       [ 0.,  9.],
       [ 7.,  0.],
       [ 9.,  1.],
       [ 9., 17.],
       [ 8., 17.],
       [ 5., 16.]])
```

It's changed to `float64` dtype, but there's still the same range:

```py
>>> vertices.ptp(axis=0)
array([ 9., 17.])
```

i.e.

```
>>> vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()
([0.0, 0.0], [9.0, 17.0])
```

So something must go wrong precisely after this...

The next step was

```py
hull_perim_r, hull_perim_c = polygon_perimeter(vertices[:, 0], vertices[:, 1])
```

where `polygon_perimeter` was a function from `skimage.draw.polygon_perimeter`
which was the first in the chain of replacements for `grid_points_in_poly`

Thankfully I already found the [docs for this function earlier]:

> - The first of those new lines sets 2 variables for the row and column of the CH perimeter,
>   [the docs](https://scikit-image.org/docs/stable/api/skimage.draw.html#skimage.draw.polygon_perimeter)
>   for which note "May be used to directly index into an array"

So based on the previous variables `coords` and `vertices`, I'd expect these to be in the range
`[0, 0]` and `[9, 17]`, and based on the docs I'm expecting the returned variables to go back to
integer dtypes when `hull_perim_r` and `hull_perim_c` are assigned.

```py
>>> hull_perim_r
array([0, 0, 0, 0, 0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 9, 9, 9, 9,
       9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 7, 6, 5, 5, 4, 3, 2,
       1, 0])
```

So far so good - the maximum here is 9 again.

```
>>> hull_perim_c
array([12, 11, 10,  9,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0,  0,  1,  1,
        1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       17, 17, 17, 17, 16, 16, 16, 15, 14, 14, 13, 12])
```

...and again, the maximum here is 17.

We're now just 2 lines away from the source of the `IndexError`...

```py
mask = np.zeros(image.shape, dtype=np.bool)
mask[hull_perim_r, hull_perim_c] = True
```

In the first of these lines, `image` should again be replaced by `img` in our testing code
as it refers to the bool-type `SAMPLE` image. The mask actually gets calculated fine, no
`IndexError`:

![](mask_convex_hull.png)

...so suddenly I can't reproduce the bug...

:thinking: :bug: :thinking: :bug: :thinking: :bug: :thinking:

I wrote some simple functions in `test_convex_area.py` that can be run to show this odd behaviour:

```sh
python -i test_convex_area.py
```

⇣

```STDOUT
bug_initial raised the IndexError index 10 is out of bounds for axis 0 with size 10
bug_secondary raised the IndexError index 10 is out of bounds for axis 0 with size 10
bug_tertiary raised the IndexError index 10 is out of bounds for axis 0 with size 10
bug_quaternary raised the IndexError index 10 is out of bounds for axis 0 with size 10
bug_5ary raised no error.
```

Adding in loops repeating it many times, and only testing `bug_5ary`, there's still no error
found (i.e. it's not the case that the bug is only happening some small percentage of the time)...

---

The sprint is almost over, and the result was surprisingly inconclusive (but that's better than getting
stuck I suppose). I learnt a lot about Cython, which I'd never seen before this weekend :—)

But it's not over yet and looking again at the results, I probably missed something... Let's look
more closely.

---

The first thing I notice I missed was `offset_coordinates`: notice that it's `True` by default

```py
def convex_hull_image(image, offset_coordinates=True, tolerance=1e-10):
    ...
    return mask
```

and because of this we execute an `if` block on line 71:

```py
# Add a vertex for the middle of each pixel edge
if offset_coordinates:
    offsets = _offsets_diamond(image.ndim)
    coords = (coords[:, np.newaxis, :] + offsets).reshape(-1, ndim)

# repeated coordinates can *sometimes* cause problems in
# scipy.spatial.ConvexHull, so we remove them.
coords = unique_rows(coords)
```

This was supposed to happen before assigning `hull = ConvexHull(coords)`, oops...

But actually, now we have a working test case and we can see at what point these intermediate steps
break the masking step. I just jumped to the end too soon.

The first step to include is to assign `offsets` and then edit `coords` using this.

The function `_offsets_diamond` is defined on lines 15-19 of `skimage/morphology/convex_hull.py`

So since there's no bug without that step, and there is a bug with this step, then this is surely
where the bug will be hiding.

...and running this:

```py
>>> (coords[:, np.newaxis, :] + offsets).reshape(-1, ndim)
```

We get quite a long array which I'll only show the first few lines of:

```py
array([[-0.5,  9. ],
       [ 0.5,  9. ],
       [ 0. ,  8.5],
       [ 0. ,  9.5],
       [ 0.5,  8. ],
       ...
       [ 7. ,  0.5],
       [ 7.5,  1. ],
       [ 8.5,  1. ],
       [ 8. ,  0.5],
       [ 8. ,  1.5],
       [ 8.5,  1. ],
       [ 9.5,  1. ],
       ...
```

and there's the source of the bug: the maximum value in the first column here (which becomes
`hull_perim_r`, the row index) has been incremented from 9 to 9.5 on the last line I showed here,
and it's this that must be getting rounded to 10 when the array is subsequently coerced back to
integer from float (as they were in the `vertices` array).

From what I've seen earlier, I'd guess that perhaps this `_offsets_diamond` function that's
introducing the bug is perhaps doing something like the 'binary dilation' step (the mathematical
morphology) when it increments by this half-integer offset followed by rounding, but that's just a
guess.

To guess at a solution, I'd suggest clipping the effect of this incrementation based on the maximum
and minimum of the array (but I'd want to make sure this makes sense once I understand the purpose
of the `_offset_diamond` function).

Once again there are two ways of showing this range:

```py
>>> (coords[:, np.newaxis, :] + offsets).reshape(-1, ndim).ptp(axis=0)
array([10., 18.])
```

or

```py
>>> (coords[:, np.newaxis, :] + offsets).reshape(-1, ndim).min(axis=0)
array([-0.5, -0.5])

>>> (coords[:, np.newaxis, :] + offsets).reshape(-1, ndim).max(axis=0)
array([ 9.5, 17.5])
```

So putting these 2 lines (and no further code) into a bug reproduction function,
`bug_5ary` (and renaming the old one that didn't error `nobug_5ary`), and running the test
over them all now we get:

```sh
python -i test_convex_area.py
```
⇣
```STDOUT
bug_initial raised the IndexError index 10 is out of bounds for axis 0 with size 10
bug_secondary raised the IndexError index 10 is out of bounds for axis 0 with size 10
bug_tertiary raised the IndexError index 10 is out of bounds for axis 0 with size 10
bug_quaternary raised the IndexError index 10 is out of bounds for axis 0 with size 10
nobug_5ary raised no error.
bug_5ary raised the IndexError index 10 is out of bounds for axis 0 with size 10
>>>
```

...and thanks to setting a parameter `rets` which will cause the function to return early and
to populate the namespace with the values it created which would cause the error, dropping into this
program interactively will give you the variables to inspect.

- e.g. You can `print(coords.max(axis=0).tolist())` and get `[9.5, 17.5]`

I'll also included a copy of the `_offsets_diamond` function here just for reference:
(which appeared on lines 15-19 of `convex_hull.py`)

```py
def _offsets_diamond(ndim):
    offsets = np.zeros((2 * ndim, ndim))
    for vertex, (axis, offset) in enumerate(product(range(ndim), (-0.5, 0.5))):
        offsets[vertex, axis] = offset
    return offsets
```

### Anything else worth fixing while we're at it

Not to be pedantic, but I also don't see any particular reason why the two lines here use
firstly `image.ndim` and then `ndim` (but it's not that important).

### Reviewing the newly discovered source of the bug

So what is this function doing, `_offsets_diamond`? The comment says:

> "Add a vertex for the middle of each pixel edge"

`TODO`: figure out how this relates to the algorithm logic, using the book references above

### Recap of how to reproduce this result

First set up your conda environment and clone the `skimage` dev repo as outlined [here]()

For me that meant

- As a first time contributor, forking the project and cloning and adding an upstream repo as
  outlined [here](https://scikit-image.org/docs/stable/contribute.html#development-process)
- following the [conda instructions](https://scikit-image.org/docs/stable/contribute.html#conda))
  so that I could `conda activate` a virtual environment dedicated to scikit-image dev
  - Additionally, I got one error, which escalated to one failed test after running

```sh
conda install pytest-httpserver
```

(If using conda you should run this in addition to the other instructions if you get an error which
mentions `pytest-localserver`)

- Fun fact, the `pip` installation takes a long time for this package because of the work it's doing
  "Cythonizing and compiling the C code" (according to Emmanuelle) !

Then to get the branch you need to checkout and pull, but first you might want to `cp -r` the entire
cloned repo directory on your local machine you just got, so that you have both an up to date but unpatched
copy of the repo as well as the one you're about to apply the patch to (this will modify the files
in the directory, so if you don't copy you will no longer have a copy up to date with the
scikit-image dev repo):

- Note that the patch the following `git pull` will download is "2537 commits behind
  scikit-image:master"! Ok that's enough warning :—P

```sh
git checkout -b ehusby-patch-1 master
git pull https://github.com/ehusby/scikit-image.git patch-1
```

Then copy the script `test_convex_area.py` from this repo into the top-level of that (`scikit-image`) repo,
(or modify it to run from somewhere else if you prefer), and you should be able to run it to reproduce the
results I report above.

### Suggesting what to do next

To review the pull request which began this whole adventure through skimage,
Erik Husby (ehusby) writes in issue 2928:

> “My hunch is that the current `offset_coordinates` keyword argument is causing most (if not all) of
> the index-out-of-bound errors, given that the `skimage.draw.polygon_perimeter` method in `faster`
> that replaces the polygon-drawing functionality of `grid_points_in_poly` in `latest` essentially
> produces the desired coordinate offset in 2D. Giving `offset_coordinates=True` to `faster`
> essentially dilates the resulting convex polygon from what is desired, which will surely cause the
> array bounds error if the polygon is to share an edge with the border of the image”

So Erik vaguely saw the source (I couldn’t really understand it before I looked at the source, it
was a bit overwhelming), note that when he refers to “faster” he’s referring to the branch `patch-1`
which his PR would create.

...My question then is should we discard the invalid dilation, or does it sometimes have meaning?
E.g. would you ever call convex hull on some sub-portion of an image, in which case would it make
sense to say that the ‘trimmed dilation’ caused by the `_offset_diamond` function should be applied
to the edges of the larger image the sub image is a part of?

My guess (as far as I can tell) is that no, you only call convex hull on an image, so it doesn’t
make sense to have ‘residual’ border incrementation to ‘add on’ to any parent image.

...but for now it's the end of the sprint and time for me to go to bed :—)

(Will figure out the next steps and arrange a response to the PR shortly)
