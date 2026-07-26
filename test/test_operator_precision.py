"""
The assembled operator must carry the precision of its inputs.

make_smooth() used to accumulate into a float32 buffer while every input
(dx, dy, elem_area, Mt) was float64 and jax_enable_x64 was on. On a uniform
mesh that is bit-identical -- every entry is a small-integer multiple of one
common float -- which is why the rest of the suite never noticed. On spherical
or non-uniform meshes it cost ~1e-7..1e-4 relative accuracy and broke the
operator's exact constant-null-space property.
"""
import warnings

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from implicit_filter.utils._auxiliary import (
    make_tri, neighboring_triangles, neighbouring_nodes, areas,
)
from implicit_filter.utils._jax_function import make_smooth, make_smat


def spherical_mesh(n=16):
    nodnum = np.reshape(np.arange(n * n), [n, n]).T
    lon = np.zeros((n, n))
    lat = np.zeros((n, n))
    for i in range(n):
        lat[i, :] = np.linspace(60.0, 70.0, n)
    for i in range(n):
        lon[:, i] = np.linspace(0.0, 20.0, n)
    tri = make_tri(nodnum, n, n)
    return n * n, len(tri), tri, lon.flatten(), lat.flatten()


def assemble(cartesian=False):
    n2d, e2d, tri, xcoord, ycoord = spherical_mesh()
    ne_num, ne_pos = neighboring_triangles(n2d, e2d, tri)
    nn_num, nn_pos = neighbouring_nodes(n2d, tri, ne_num, ne_pos)
    area, elem_area, dx, dy, Mt = areas(
        n2d, e2d, tri, xcoord, ycoord, ne_num, ne_pos,
        "r", cartesian, 2 * np.pi, np.ones(e2d),
    )
    smooth, metric = make_smooth(
        jnp.array(Mt), jnp.array(elem_area), jnp.array(dx), jnp.array(dy),
        jnp.array(nn_num), jnp.array(nn_pos), jnp.array(tri), n2d, e2d, False,
    )
    return smooth, metric, nn_num, nn_pos, n2d


class TestAssemblyDtype:
    def test_x64_is_enabled(self):
        """Guard: the whole point is that the package asks for float64."""
        assert jax.config.jax_enable_x64 is True

    def test_smooth_matrix_is_double_precision(self):
        smooth, _, _, _, _ = assemble()
        assert smooth.dtype == jnp.float64, (
            f"operator assembled in {smooth.dtype} while inputs are float64"
        )

    def test_metric_matrix_is_double_precision(self):
        _, metric, _, _, _ = assemble()
        assert metric.dtype == jnp.float64

    def test_no_unsafe_cast_warning(self):
        """JAX warns that this cast will become a hard error."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assemble()
        offending = [
            w for w in caught
            if "incompatible types" in str(w.message)
            or "cannot safely cast" in str(w.message)
        ]
        assert not offending, (
            "JAX reported an unsafe dtype cast during assembly: "
            f"{[str(w.message)[:160] for w in offending]}"
        )


class TestConstantNullSpace:
    """A Laplacian must annihilate constants to machine precision."""

    def test_row_sums_vanish_to_double_precision(self):
        smooth, _, nn_num, nn_pos, n2d = assemble()
        nza = int(np.sum(nn_num))
        ss, ii, jj = make_smat(
            jnp.array(nn_pos), jnp.array(nn_num), smooth, n2d, nza
        )
        row_sums = np.zeros(n2d)
        np.add.at(row_sums, np.asarray(ii), np.asarray(ss))

        scale = float(np.max(np.abs(np.asarray(ss))))
        worst = float(np.max(np.abs(row_sums))) / scale
        assert worst < 1e-12, (
            f"relative row-sum residual {worst:.3e} indicates the operator was "
            "accumulated at reduced precision"
        )
