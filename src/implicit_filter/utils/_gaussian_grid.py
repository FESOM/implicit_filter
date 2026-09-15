"""
Reduced Gaussian grids (ECMWF): Gaussian latitudes, the classical ``N`` and
octahedral ``O`` grid definitions, and a Delaunay triangulation of points
covering the whole sphere.

Points are generated in ECMWF storage order -- latitude rows from north to
south, each row from west to east starting at 0 degrees longitude -- which is
the order of the ``values`` dimension of a GRIB message and of the public
ERA5 zarr stores.
"""
import operator
import re

import numpy as np

from ._reduced_gaussian_tables import CLASSICAL_PL

_GRID_NAME = re.compile(r"^\s*([NnOo])(\d+)\s*$")

#: How much longer a triangle's longest edge may be than the smaller of its
#: vertices' local (median incident-edge) length scales before
#: ``spherical_triangulation`` treats it as bridging a hole rather than a
#: legitimate mesh triangle. On every bundled ECMWF reduced Gaussian grid
#: the widest legitimate triangles are those on the polar-most rows (18-20
#: points), and they measure only 2.0-2.2 -- a factor of about 3.7 below
#: this threshold. The bound *rises* with the number of points on an
#: isotropic polar ring: qhull closes a co-circular ring of ``n`` points
#: with a single-vertex fan, and the triangles at the low-degree ring nodes
#: beside the fan hub measure about ``1 / sin(pi / n)`` -- 7.66 at n = 24,
#: crossing 8.0 between n = 25 and n = 26 (a synthetic isotropic ring measures
#: a little below the formula: 7.48 at n = 24, 8.13 at n = 26). A global grid
#: with a larger isotropic polar-most ring would therefore need a larger
#: factor. Grids whose polar ring is dense and anisotropic -- regular lat-lon
#: grids, or full (non-reduced) Gaussian ``F`` grids -- are rejected by default,
#: and deliberately so: the ring node beside the fan hub keeps only degree 3
#: (two zonal edges plus one meridional), so its median incident edge is the
#: zonal spacing and the strip triangles touching it measure about 46 on a
#: 2.5 degree lat-lon grid. Use ``LatLonFilter`` for such a regular grid, or
#: pass ``check_coverage=False`` to skip the heuristic.
_HOLE_FACTOR = 8.0


def _check_n(N) -> int:
    """
    Validate a Gaussian-grid ``N`` parameter and return it as a plain ``int``.

    Raises
    ------
    TypeError
        If ``N`` is not an integer (``bool`` included -- it is an ``int``
        subclass that would otherwise silently pass ``operator.index``).
    ValueError
        If ``N`` is not positive.
    """
    if isinstance(N, bool):
        raise TypeError(f"N must be an integer, got a bool ({N!r})")
    N = operator.index(N)
    if N < 1:
        raise ValueError(f"N must be a positive integer, got {N}")
    return N


def gaussian_latitudes(N: int) -> np.ndarray:
    """
    The ``2N`` Gaussian latitudes of an ``N`` grid, in degrees, north to south.

    These are the roots of the Legendre polynomial ``P_2N``, found by Newton
    iteration on the three-term recurrence (machine precision; agrees with
    ecCodes' ``codes_get_gaussian_latitudes`` to better than 1e-11 degrees).

    Parameters
    ----------
    N : int
        Number of latitude rows between a pole and the equator.

    Returns
    -------
    np.ndarray
        Shape ``(2N,)``, strictly decreasing, symmetric about the equator.
    """
    N = _check_n(N)
    n = 2 * N
    i = np.arange(1, N + 1)
    x = np.cos(np.pi * (i - 0.25) / (n + 0.5))    # roots in the northern half
    for _ in range(100):
        p0 = np.ones_like(x)
        p1 = x.copy()
        for k in range(2, n + 1):
            p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
        dp = n * (x * p1 - p0) / (x * x - 1.0)     # derivative of P_n
        dx = p1 / dp
        x = x - dx
        if np.max(np.abs(dx)) < 1e-15:
            break
    else:
        raise RuntimeError("Gauss-Legendre root finding did not converge")
    lat = np.degrees(np.arcsin(x))
    return np.concatenate([lat, -lat[::-1]])


def octahedral_pl(N: int) -> np.ndarray:
    """
    Points per latitude row of the octahedral grid ``O<N>``, north to south.

    Row ``j`` (1-based from the pole) has ``4 j + 16`` points, so the grid has
    ``4 N (N + 9)`` points in total (O96: 40 320).
    """
    N = _check_n(N)
    half = 20 + 4 * np.arange(N)
    return np.concatenate([half, half[::-1]])


def classical_pl(N: int) -> np.ndarray:
    """
    Points per latitude row of the classical reduced Gaussian grid ``N<N>``,
    north to south, from the tables ECMWF defines (see
    ``_reduced_gaussian_tables``).

    Raises
    ------
    ValueError
        If no table is bundled for this ``N``.
    """
    N = _check_n(N)
    try:
        half = np.asarray(CLASSICAL_PL[N], dtype=int)
    except KeyError:
        raise ValueError(
            f"No table for the classical reduced Gaussian grid N{N}; bundled "
            f"grids: {', '.join('N%d' % n for n in sorted(CLASSICAL_PL))}. "
            "Octahedral grids (O<N>) are available for any N. For other grids "
            "build the filter from the coordinates stored in your data file "
            "(ReducedGaussianFilter.prepare_from_points / prepare_from_data_array)."
        ) from None
    return np.concatenate([half, half[::-1]])


def reduced_gaussian_grid(name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Coordinates of every point of an ECMWF reduced Gaussian grid.

    Parameters
    ----------
    name : str
        Grid name: ``'O<N>'`` for octahedral grids (e.g. ``'O96'``,
        ``'O1280'``) or ``'N<N>'`` for classical grids (e.g. ``'N320'``).
        Case-insensitive.

    Returns
    -------
    (lat, lon) : tuple[np.ndarray, np.ndarray]
        Latitudes and longitudes in degrees, one entry per grid point, in
        ECMWF storage order: rows north to south, each row west to east
        starting at 0 degrees. Longitudes are in ``[0, 360)``.
    """
    m = _GRID_NAME.match(str(name))
    if not m:
        raise ValueError(
            f"Unrecognised reduced Gaussian grid name {name!r}; expected "
            "'O<N>' (octahedral, e.g. 'O96') or 'N<N>' (classical, e.g. 'N320')"
        )
    kind, N = m.group(1).upper(), int(m.group(2))
    pl = octahedral_pl(N) if kind == "O" else classical_pl(N)
    lat = np.repeat(gaussian_latitudes(N), pl)
    lon = np.concatenate([np.arange(p) * (360.0 / p) for p in pl])
    return lat, lon


def spherical_triangulation(lat, lon, check_coverage: bool = True) -> np.ndarray:
    """
    Delaunay triangulation of points covering the whole sphere.

    The Delaunay triangulation of points on a sphere is the convex hull of
    their unit vectors, so this uses ``scipy.spatial.ConvexHull``. The result
    is a closed triangulated surface with ``2n - 4`` triangles. A point set
    counts as covering the sphere when both hold: the origin lies strictly
    inside the hull (rules out regional/hemispherical data; exact), and --
    when ``check_coverage`` is true -- no triangle has an edge longer than
    ``_HOLE_FACTOR`` (8x) the local median incident-edge length at its
    vertices (rules out point sets that wrap around the origin but leave a
    hole, e.g. a masked-out polar cap or land/ocean band; a heuristic, not
    exact -- see below).

    The hole heuristic flags any triangle whose longest edge exceeds 8x the
    median length of the edges meeting at one of its vertices. It is tuned
    for quasi-uniform point sets such as ECMWF's reduced Gaussian grids,
    whose polar rings have 18-20 points, and as a consequence it rejects
    some point sets that do cover the sphere: any with a dense polar ring
    that has many more points than its neighbouring rows -- regular
    lat-lon-like grids and full (non-reduced) Gaussian ``F`` grids -- or,
    more generally, an isotropic polar ring of more than about 24 points.
    Grids like that are regular and their home in this package is
    ``LatLonFilter``; pass ``check_coverage=False`` to skip the heuristic
    (the exact checks -- shape, convex hull, origin-inside, duplicate
    points -- always run) if a triangulation from this function is still
    wanted for one.

    Parameters
    ----------
    lat, lon : array_like
        1-D coordinates in degrees, any longitude convention.
    check_coverage : bool, optional
        Whether to run the hole heuristic described above, in addition to
        the exact shape/hull/origin/duplicate checks, which always run.
        Default ``True``. Set to ``False`` for a point set the heuristic
        rejects despite being complete, e.g. a regular lat-lon or full
        Gaussian grid with a dense polar ring.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(2n - 4, 3)`` with node indices into the
        input arrays. Triangle orientation is arbitrary (the nodal filter
        does not depend on it).

    Raises
    ------
    ValueError
        If the coordinates are not 1-D arrays of equal length, if the points
        do not cover the sphere -- the origin is not strictly inside their
        convex hull, or (when ``check_coverage`` is true) the hull has to
        bridge a hole in the point set (regional or masked data -- use
        ``LatLonFilter`` or ``TriangularFilter`` with your own
        triangulation), or if they contain duplicate points. The hole check
        is a coarse, local test: gaps narrower than about eight local point
        spacings -- on O32 that still includes a missing 15-degree-wide
        longitude band, or the three northernmost rows (lat > 80 degrees) --
        are not detected and the triangulation is returned as if the data
        were complete. Conversely, it rejects some complete point sets
        outright: a dense, quasi-isotropic polar ring of more than about 24
        points (regular lat-lon-like and full Gaussian ``F`` grids) --
        pass ``check_coverage=False`` for those, or use ``LatLonFilter``.
    """
    from scipy.spatial import ConvexHull, QhullError

    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if lat.ndim != 1 or lat.shape != lon.shape:
        raise ValueError(
            f"lat and lon must be 1-D arrays of equal length, got shapes "
            f"{lat.shape} and {lon.shape}")
    n = lat.size
    if n < 4:
        raise ValueError("at least 4 points are needed to triangulate a sphere")
    la = np.radians(lat)
    lo = np.radians(lon)
    xyz = np.column_stack(
        [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    try:
        hull = ConvexHull(xyz)
    except QhullError as exc:
        raise ValueError(
            "Could not triangulate the points on the sphere (degenerate input, "
            "e.g. all points on one great circle or many coincident points): "
            f"{exc}") from exc
    # Outward facet normals: a point is inside where normal . x + offset < 0.
    # The origin is strictly inside the hull only if the points surround it,
    # i.e. cover the sphere; a regional point set closes its hull with facets
    # that pass between the origin and the data.
    if np.any(hull.equations[:, -1] > -1e-12):
        raise ValueError(
            "The points do not cover the whole sphere (the origin is not "
            "strictly inside their convex hull), so no global triangulation "
            "exists. For regional data use LatLonFilter, or TriangularFilter "
            "with your own triangulation.")
    tri = hull.simplices.astype(int)
    node_count = np.unique(tri).size
    if len(tri) != 2 * n - 4 or node_count != n:
        raise ValueError(
            f"Expected {2 * n - 4} triangles using all {n} points but got "
            f"{len(tri)} triangles on {node_count} points: the input "
            "contains duplicate (coincident) points.")
    if check_coverage:
        # The origin-inside-hull check above is necessary but not sufficient:
        # a point set that wraps around the origin while missing a region
        # (e.g. a masked-out polar cap or a continent-sized longitude band)
        # still passes it, and the hull silently closes the gap with
        # triangles that bridge it. Flag any triangle whose longest edge is
        # far longer than the local length scale at its vertices -- such
        # triangles only arise when the hull has to span a hole. The local
        # scale is the median length of the triangulation edges meeting at
        # that node (not the nearest-neighbour distance): that is robust to
        # one anomalously short edge, e.g. a near-duplicate point, which
        # would otherwise collapse the scale for every triangle that touches
        # it, and to anisotropic spacing (zonal much smaller than meridional
        # on the polar rows of lat-lon and full Gaussian grids), where the
        # nearest-neighbour distance is not the cell size. It is still a
        # heuristic tuned for quasi-uniform grids (see the docstring): a
        # dense, quasi-isotropic polar ring -- as in a regular lat-lon or
        # full Gaussian grid -- trips it even though the grid is complete;
        # pass check_coverage=False to skip this block for such grids.
        e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
        e.sort(axis=1)
        e = np.unique(e[:, 0].astype(np.int64) * n + e[:, 1])      # each edge once
        e = np.column_stack([e // n, e % n])
        length = np.linalg.norm(xyz[e[:, 0]] - xyz[e[:, 1]], axis=1)
        node = np.concatenate([e[:, 0], e[:, 1]])
        node_len = np.concatenate([length, length])
        order = np.lexsort((node_len, node))
        node, node_len = node[order], node_len[order]
        count = np.bincount(node, minlength=n)  # >= 3 on a closed surface
        start = np.concatenate([[0], np.cumsum(count)[:-1]])
        scale = node_len[start + count // 2]    # upper median per node
        longest = np.max(np.stack(
            [np.linalg.norm(xyz[tri[:, a]] - xyz[tri[:, b]], axis=1)
             for a, b in ((0, 1), (1, 2), (2, 0))]), axis=0)
        local_scale = np.min(scale[tri], axis=1)
        bad = longest > _HOLE_FACTOR * local_scale
        if np.any(bad):
            n_bad = int(np.sum(bad))
            worst_ratio = float(np.max(longest[bad] / local_scale[bad]))
            raise ValueError(
                f"The points do not cover the whole sphere: {n_bad} triangle(s) "
                f"span a gap much wider than the local point spacing (largest "
                f"edge {worst_ratio:.1f} times the local median edge length). "
                "Regional or masked data cannot be triangulated globally; pass "
                "the full grid and use the mask= argument of "
                "ReducedGaussianFilter instead, or use LatLonFilter / "
                "TriangularFilter with your own triangulation. If the points "
                "really do cover the sphere but include a dense polar ring "
                "(regular lat-lon or full Gaussian grids), use LatLonFilter "
                "for that regular grid, or pass check_coverage=False to skip "
                "this heuristic.")
    return tri
