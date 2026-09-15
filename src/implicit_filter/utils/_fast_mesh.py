"""
Vectorised NumPy replacements for the pure-Python mesh loops in ``_auxiliary``
(``neighboring_triangles``, ``neighbouring_nodes`` and ``areas``), which
dominate ``TriangularFilter.prepare`` on large meshes. The Python versions
stay in ``_auxiliary`` as the reference implementations; the functions here
reproduce their results bit for bit -- the same neighbour ordering and the
same floating-point operation order -- which ``test/test_fast_mesh.py`` pins
on several meshes, so that no prepared filter changes by a single bit.

NumPy rather than JAX is used deliberately: these are index/sort operations
and per-triangle scalar arithmetic, where NumPy is already memory-bound, and
XLA may fuse multiply-adds or reorder scatter-adds, which would break the
bit-identity with the reference. The JAX assembly (``make_smooth``) that
follows in ``prepare`` is unchanged.
"""
import math

import numpy as np

from . import _auxiliary
from ._auxiliary import tangent_plane_geometry

#: NumPy switches to a recursive pairwise scheme above this many elements.
#: Mesh node valences stay far below it -- 19 on N320, 21 on the O grids --
#: except for the polar fans of a dense regular lat/lon point set, which
#: ``fast_areas`` sums per node instead.
_NUMPY_PAIRWISE_BLOCK = 128

#: NumPy 2 introduced NEP 50 promotion. Under 1.x the reference's scalar
#: arithmetic promotes by value and widens to float64 where the vectorised
#: arrays here stay float32, so non-float64 meshes are delegated there.
_NUMPY_MAJOR = int(np.__version__.split(".")[0])


def fast_neighboring_triangles(n2d: int, e2d: int, tri: np.ndarray):
    """
    Vectorised ``neighboring_triangles``: identical ``ne_num`` and ``ne_pos``.

    For every node the adjacent triangles are listed in increasing triangle
    index; a triangle that repeats a node index is listed once for that node,
    exactly as the loop reference does (its fancy-indexed ``+= 1`` counts a
    repeated index once and its slot write is overwritten in place).
    """
    tri = np.asarray(tri)[:e2d]
    # np.intp: np.bincount below refuses uint64, which the reference accepts
    nodes = tri.ravel().astype(np.intp, copy=False)      # triangle-major, vertex order
    tris = np.repeat(np.arange(e2d), 3)
    order = np.argsort(nodes, kind="stable")             # group by node, keep triangle order
    nodes, tris = nodes[order], tris[order]
    keep = np.ones(nodes.size, dtype=bool)               # drop repeated (node, triangle) pairs
    keep[1:] = (nodes[1:] != nodes[:-1]) | (tris[1:] != tris[:-1])
    nodes, tris = nodes[keep], tris[keep]
    ne_num = np.bincount(nodes, minlength=n2d).astype(int)
    start = np.concatenate(([0], np.cumsum(ne_num)[:-1]))
    rank = np.arange(nodes.size) - start[nodes]
    ne_pos = np.zeros([int(np.max(ne_num)), n2d], dtype=int)
    ne_pos[rank, nodes] = tris
    return ne_num, ne_pos


def fast_neighbouring_nodes(n2d: int, tri: np.ndarray, ne_num: np.ndarray, ne_pos: np.ndarray):
    """
    Vectorised ``neighbouring_nodes``: identical ``nn_num`` and ``nn_pos``.

    The reference scans, for node ``j``, the vertices of its adjacent
    triangles (triangles in ``ne_pos`` order, vertices in row order) and keeps
    every node in order of first appearance -- ``j`` itself included. This
    reproduces that order by sorting the scan by (node, candidate, position),
    keeping the first position of each pair, and sorting back by position.
    """
    tri = np.asarray(tri)
    ne_num = np.asarray(ne_num)
    total = int(ne_num.sum())
    node = np.repeat(np.arange(n2d), ne_num)             # scan order: node-major ...
    start = np.concatenate(([0], np.cumsum(ne_num)[:-1]))
    elem = np.asarray(ne_pos)[np.arange(total) - start[node], node]   # ... triangles in ne_pos order
    j = np.repeat(node, 3)                                # ... vertices in row order
    a = tri[elem].ravel().astype(np.intp, copy=False)      # np.intp: see above
    seq = np.arange(a.size)
    order = np.lexsort((seq, a, j))                       # by node, candidate, position
    j_s, a_s, seq_s = j[order], a[order], seq[order]
    first = np.ones(a.size, dtype=bool)
    first[1:] = (j_s[1:] != j_s[:-1]) | (a_s[1:] != a_s[:-1])
    j_f, a_f, seq_f = j_s[first], a_s[first], seq_s[first]
    back = np.lexsort((seq_f, j_f))                       # scan order within each node
    j_f, a_f = j_f[back], a_f[back]
    nn_num = np.bincount(j_f, minlength=n2d).astype(int)
    start = np.concatenate(([0], np.cumsum(nn_num)[:-1]))
    rank = np.arange(j_f.size) - start[j_f]
    nn_pos = np.zeros([int(np.max(nn_num)), n2d], dtype=int)
    nn_pos[rank, j_f] = a_f
    return nn_num, nn_pos


def pairwise_row_sums(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """
    Row sums in NumPy's own summation order.

    ``values`` is ``(rows, M)`` with row ``r`` meaningful in its first
    ``counts[r]`` entries and zero beyond. Returns exactly what
    ``np.sum(values[r, :counts[r]])`` returns for every row: NumPy adds fewer
    than 8 elements sequentially starting from 0.0; otherwise it seeds 8
    accumulators with the first 8 elements, adds every further complete block
    of 8 into them, combines them as ``((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))``
    and adds the remaining tail sequentially. Node areas depend on this order
    in the last bit, so the reference loop's ``np.sum`` per node is
    reproduced here for all nodes at once. Adding the zero padding is exact,
    so only the association structure has to be mimicked.
    """
    values = np.asarray(values, dtype=float)
    counts = np.asarray(counts)
    rows, M = values.shape
    if M > _NUMPY_PAIRWISE_BLOCK:
        raise NotImplementedError(
            f"rows of more than {_NUMPY_PAIRWISE_BLOCK} elements would use NumPy's "
            "recursive pairwise summation, which is not reproduced here")

    res_seq = np.zeros(rows)                              # fewer than 8 elements
    for i in range(min(M, 7)):
        res_seq = res_seq + np.where(counts > i, values[:, i], 0.0)
    if M < 8:
        return res_seq

    r = [values[:, i].copy() for i in range(8)]           # 8 accumulators seeded
    i = 8
    while i + 8 <= M:                                     # complete blocks only
        block = counts >= i + 8
        for k in range(8):
            r[k] = r[k] + np.where(block, values[:, i + k], 0.0)
        i += 8
    res_pw = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
    tail_start = counts - counts % 8
    for i in range(8, M):                                 # sequential tail
        tail = (i >= tail_start) & (i < counts)
        res_pw = res_pw + np.where(tail, values[:, i], 0.0)
    return np.where(counts < 8, res_seq, res_pw)


def fast_areas(
    n2d, e2d, tri, xcoord, ycoord, ne_num, ne_pos, meshtype, carthesian, cyclic_length, mask
):
    """
    Vectorised ``areas``: identical ``area``, ``elem_area``, ``dx``, ``dy``,
    ``Mt``. Every expression below is the reference's, evaluated for all
    triangles at once in the same operation order; node areas use
    :func:`pairwise_row_sums` to match ``np.sum`` per node.

    The coordinates keep the dtype they are given. The reference indexes them
    as they are, so with float32 input the whole per-triangle chain runs in
    single precision (and so must this one); only the containers it stores
    into are float64, which the widening cast after the branches reproduces.
    """
    # Corner cases of NumPy's type promotion, handed to the reference loop,
    # which is exact by definition. They can only bite when the coordinates are
    # not float64: then the reference's per-element *scalar* arithmetic and the
    # *array* arithmetic here may promote differently. 's' is unaffected
    # (tangent_plane_geometry casts to float64 on both paths) and the unknown
    # meshtype produces zeros, so only 'm' and 'r' are guarded.
    if meshtype in ("m", "r") and (
        np.asarray(xcoord).dtype != np.float64
        or np.asarray(ycoord).dtype != np.float64
    ) and (
        # NumPy 1.x promotes by value, widening the reference's scalar chain
        # to float64 while these arrays stay float32.
        _NUMPY_MAJOR < 2
        # A NumPy-typed cyclic_length is strongly typed, so `x2 - cyclic_length`
        # is float64 here for every element, whereas the reference rebinds only
        # the elements whose cyclic branch actually fires.
        or isinstance(cyclic_length, (np.generic, np.ndarray))
        # An integer mask is a weak Python int per element in the reference
        # (`mask[n]` of a list) but a strongly typed array here.
        or np.asarray(mask).dtype.kind not in "bf"
    ):
        return _auxiliary.areas(
            n2d, e2d, tri, xcoord, ycoord, ne_num, ne_pos, meshtype,
            carthesian, cyclic_length, mask)

    tri = np.asarray(tri)
    xcoord = np.asarray(xcoord)         # no dtype cast: see the docstring
    ycoord = np.asarray(ycoord)
    mask = np.asarray(mask)
    ne_num = np.asarray(ne_num)
    ne_pos = np.asarray(ne_pos)
    r_earth = _auxiliary.R_EARTH        # read at call time: single source of truth, patchable
    t = tri[:e2d]
    mask_e = mask[:e2d]                 # the reference only reads mask[n], n < e2d
    Mt = np.ones([e2d])

    if meshtype == "m":
        x2 = xcoord[t[:, 1]] - xcoord[t[:, 0]]
        x3 = xcoord[t[:, 2]] - xcoord[t[:, 0]]
        y2 = ycoord[t[:, 1]] - ycoord[t[:, 0]]
        y3 = ycoord[t[:, 2]] - ycoord[t[:, 0]]
        d = x2 * y3 - y2 * x3
        dx = np.column_stack([(-y3 + y2) / d, y3 / d, -y2 / d])
        dy = np.column_stack([-(-x3 + x2) / d, -x3 / d, x2 / d])
        elem_area = 0.5 * np.abs(d) * mask_e

    elif meshtype == "r":
        rad = math.pi / 180.0
        if carthesian:
            Mt = np.ones([e2d])
        else:
            Mt = np.cos(np.sum(rad * ycoord[tri], axis=1) / 3.0)
        x2 = rad * (xcoord[t[:, 1]] - xcoord[t[:, 0]])
        x3 = rad * (xcoord[t[:, 2]] - xcoord[t[:, 0]])
        y2 = r_earth * rad * (ycoord[t[:, 1]] - ycoord[t[:, 0]])
        y3 = r_earth * rad * (ycoord[t[:, 2]] - ycoord[t[:, 0]])
        # cyclic corrections, applied one after the other as in the reference
        x2 = np.where(x2 > cyclic_length / 2.0, x2 - cyclic_length, x2)
        x2 = np.where(x2 < -cyclic_length / 2.0, x2 + cyclic_length, x2)
        x3 = np.where(x3 > cyclic_length / 2.0, x3 - cyclic_length, x3)
        x3 = np.where(x3 < -cyclic_length / 2.0, x3 + cyclic_length, x3)
        x2 = r_earth * x2 * Mt[:e2d]
        x3 = r_earth * x3 * Mt[:e2d]
        d = x2 * y3 - y2 * x3
        dx = np.column_stack([(-y3 + y2) / d, y3 / d, -y2 / d])
        dy = np.column_stack([-(-x3 + x2) / d, -x3 / d, x2 / d])
        elem_area = 0.5 * np.abs(d) * mask_e
        if carthesian:
            Mt = np.zeros([e2d])
        else:
            Mt = (np.sin(rad * np.sum(ycoord[tri], axis=1) / 3.0) / Mt) / r_earth

    elif meshtype == "s":
        elem_area, dx, dy, Mt = tangent_plane_geometry(tri, xcoord, ycoord, mask)

    else:                               # mirror the reference: no geometry at all
        dx = np.zeros([e2d, 3], dtype=float)
        dy = np.zeros([e2d, 3], dtype=float)
        elem_area = np.zeros([e2d])

    # The reference computes in the coordinates' precision but stores into
    # float64 containers (np.zeros([e2d, 3], dtype=float), np.zeros([e2d])),
    # so widen here. Exact, and a no-op when the chain was float64 already.
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    elem_area = np.asarray(elem_area, dtype=float)

    # Node ("cluster") areas: one third of the adjacent element areas, summed
    # in ne_pos order with NumPy's summation order. The dense gather is capped
    # at _NUMPY_PAIRWISE_BLOCK columns and zeroed in place, so it costs one
    # (n2d, min(max valence, 128)) float64 buffer plus the boolean slot mask,
    # whatever the largest valence is; the few nodes above that cap take the
    # per-node branch below anyway.
    k = min(ne_pos.shape[0], _NUMPY_PAIRWISE_BLOCK)
    beyond = np.arange(k)[None, :] >= ne_num[:, None]     # the padding slots
    gathered = elem_area[ne_pos[:k].T]
    gathered[beyond] = 0.0
    if ne_pos.shape[0] <= _NUMPY_PAIRWISE_BLOCK:
        total = pairwise_row_sums(gathered, ne_num)
    else:
        # A node with more than a block of adjacent triangles is summed by
        # NumPy's recursive scheme, which pairwise_row_sums does not
        # reproduce. Such nodes are rare -- the polar fans of a regular
        # lat/lon point set have a few hundred, reduced Gaussian grids stay
        # below twenty -- so they fall back to the reference's own expression;
        # every other node stays vectorised.
        big = ne_num > _NUMPY_PAIRWISE_BLOCK
        total = pairwise_row_sums(gathered, np.where(big, 0, ne_num))
        for n in np.flatnonzero(big):
            total[n] = np.sum(elem_area[ne_pos[0 : ne_num[n], n]])
    area = total / 3.0
    area[area == 0.0] = 1.0e-5
    return area, elem_area, dx, dy, Mt
