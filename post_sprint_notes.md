In [`sprint_notes.md`](sprint_notes.md) I gave a detailed description of how I tracked down
the source of the bug preventing the faster convex hull retrieval function from being added
to `scikit-image`.

It's quite heavy on the detail, and this makes it hard to think clearly about the pertinent
part, so next I am going to solely discuss the specific bug identified as taking place
within the function `_offset_diamond`.

## Simplify the test case into a pass and a fail

Firstly, I copied `test_convex_area.py` to `simple_test_convex_area.py` and discarded the
bug reproducing functions except the final ones: `no_bug_5ary` and `bug_5ary`, renamed to
`no_bug` and `bug`, moved the `SAMPLE` data variable to a helper module `sample_data.py`,
and removed the imports that were then no longer in use.

## Refactor the test cases into subroutines and DRY

Secondly, I refactored all the common subroutines into functions `common_subroutine_1`
and `common_subroutine_2`, which has the side benefit of clarifying where the subroutines
modify the value of their inputs (in which case the function returns the inputs to be
reassigned when returning from the subroutine).

In fact, refactor the `common_subroutine_1` out of the bug functions entirely (as it
really is a mechanism for providing a default argument by proxy), and instead set the
default argument in the main function which then passes the default arguments into the
bug functions.

## Refactor any other unnecessary parts

I removed the assignment of `ndim` which seemed to be a careful approach in case `img.ndim`
changed, but it wasn't, so I just access `img.ndim` after passing the same value to
`_offsets_diamond`.

- Note that the subroutines really help clarify the logic blocks here and what is/isn't changed
  from one part of the code to the next by simply reading the arguments to and assignments from
  the subroutines
  - (as in this program all state changes are direct/procedural not stored/connected through
    objects, inheritance, side effects, etc.).

The `nobug` function doesn't need the `if rets` part, as it's only used to control the `bug`.

---

## Re-examining the source of the IndexError

The main target of this investigation was an `IndexError`, and in the previous step I honed in
on the following 2 lines in `bug`:

```py
offsets = _offsets_diamond(img.ndim)
coords = (coords[:, np.newaxis, :] + offsets).reshape(-1, img.ndim)
```

- `offsets` is a numpy array of "jitter" (or something similar), i.e. it provides a little
  perturbation of +/- half a pixel to one of the two coordinates at a time (i.e. 4 sub-pixels).

- `coords` is reassigned, by broadcasting this 4-row-by-2-column array with each row (coordinate pair)
  of the input coordinates
  - which are all discrete, i.e. it "jitters" the discrete integer-valued pixels to half-pixels in
    all possible directions [one coordinate only at each time]

This isn't what causes the `IndexError` though, that happens when these `coords`, after being
rounded up and down [thereby sending the half-pixel jitter a full pixel away] are used to index
a mask, which was created by zeroing an array with the same shape as the input image (`img`).

When this happens, the `IndexError` is specifically for the value 10, which came from the 9.5
rounding, i.e. from the `[0,9]` + `[0,0.5]` --> `[0,9.5]` --> `[0,10]`

- It's not clear whether there's also a problem due to the 0 in `[0,9]` being rounded down to
  `[-1,9]` by the `[-0.5, 0]`: perhaps this could mistakenly be accessing the last element of the
  dimension (`x[-1]` indexes the last element of `x`).
