"""
Mesh topology tests: node valence limits.

Unstructured meshes (FESOM in particular) contain nodes with high valence.
neighbouring_nodes() must cope with them rather than failing with an opaque
IndexError from a fixed-size scratch buffer.
"""
import numpy as np
import pytest

from implicit_filter.utils._auxiliary import (
    neighboring_triangles,
    neighbouring_nodes,
)


def fan_mesh(n_ring):
    """A central node surrounded by a closed ring of n_ring triangles.

    The central node ends up with n_ring neighbouring nodes plus itself.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n_ring, endpoint=False)
    xcoord = np.concatenate(([0.0], np.cos(angles)))
    ycoord = np.concatenate(([0.0], np.sin(angles)))

    tri = []
    for i in range(1, n_ring + 1):
        nxt = i + 1 if i < n_ring else 1
        tri.append([0, i, nxt])
    tri = np.array(tri, dtype=int)

    n2d = n_ring + 1
    e2d = len(tri)
    return n2d, e2d, tri, xcoord, ycoord


@pytest.mark.parametrize("n_ring", [6, 12, 19, 20, 25, 40])
def test_high_valence_node_supported(n_ring):
    """A node touching n_ring triangles must be handled for any n_ring."""
    n2d, e2d, tri, _, _ = fan_mesh(n_ring)
    ne_num, ne_pos = neighboring_triangles(n2d, e2d, tri)

    nn_num, nn_pos = neighbouring_nodes(n2d, tri, ne_num, ne_pos)

    # centre node neighbours every ring node, plus itself
    assert nn_num[0] == n_ring + 1
    assert nn_pos.shape[0] >= nn_num.max()

    # every recorded neighbour must be a real node index
    for j in range(n2d):
        nbrs = nn_pos[: nn_num[j], j]
        assert np.all(nbrs >= 0) and np.all(nbrs < n2d)
        assert len(np.unique(nbrs)) == nn_num[j], "neighbour list has duplicates"
        assert j in nbrs, "a node must be its own neighbour"


def test_neighbour_sets_are_symmetric():
    """If a is a neighbour of b then b is a neighbour of a."""
    n2d, e2d, tri, _, _ = fan_mesh(25)
    ne_num, ne_pos = neighboring_triangles(n2d, e2d, tri)
    nn_num, nn_pos = neighbouring_nodes(n2d, tri, ne_num, ne_pos)

    sets = {j: set(nn_pos[: nn_num[j], j].tolist()) for j in range(n2d)}
    for j, nbrs in sets.items():
        for k in nbrs:
            assert j in sets[k], f"{k} lists {j}? asymmetric neighbour sets"
