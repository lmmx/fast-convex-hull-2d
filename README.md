# fast-conv-hull-2d

:running: SciPy 2020 sprint for scikit-image for a faster 2D convex hull algorithm :running:

## Intro: choosing an algorithm

One fast algorithm for computing the convex hull in 2D is known as "Graham's scan", published at Bell Labs

- R. L. Graham (1972) [An Efficient Algorithm for Determining the Convex Hull of a Finite Planar Set](http://www.math.ucsd.edu/~ronspubs/72_10_convex_hull.pdf)

...another algorithm was implemented by Graham's colleague, [Kenneth L. Clarkson](https://en.wikipedia.org/wiki/Kenneth_L._Clarkson),
available from his old Bell Labs website ([archived here](https://web.archive.org/web/19980715014112/http://cm.bell-labs.com/cm/cs/who/clarkson/2dch.c)),
- (“known for his research in computational geometry… co-editor-in-chief of the _Journal of Computational Geometry_”).
- The code was listed [on his profile](https://web.archive.org/web/20081024042432/http://cm.bell-labs.com/who/clarkson/)
  as “a short, complete planar convex hull code”.
- It is included here (along with the license header) as [`2dch.c`](2dch.c)
