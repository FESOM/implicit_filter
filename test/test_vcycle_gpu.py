"""CPU vs GPU parity of the CG solve — skipped when no GPU is visible.

JAX resolves its platform set once per process: after the first array
operation, ``set_backend`` cannot move computation anymore. Each leg
therefore runs in its own subprocess (the GPU leg asserts it really got a
CUDA device), and GPU availability is probed with ``nvidia-smi`` so the
parent process never initializes JAX at all.
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest


def _nvidia_gpu_visible():
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return False
    try:
        return subprocess.run([smi, "-L"], capture_output=True,
                              timeout=30).returncode == 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _nvidia_gpu_visible(),
                                reason="no GPU visible")

_LEG = r"""
import math
import sys

import numpy as np

platform, precond, out = sys.argv[1], sys.argv[2], sys.argv[3]
from implicit_filter import TriangularFilter

f = TriangularFilter()
# JAX fixes its platform set on first array use; prepare() creates arrays,
# so the backend must be selected before it.
f.set_backend(platform)
import jax
if platform == "gpu":
    assert any(d.platform == "gpu" for d in jax.devices()), jax.devices()

xs, ys = np.meshgrid(np.arange(41) * 5.0, np.arange(41) * 5.0)   # 1681 nodes
x, y = xs.ravel(), ys.ravel()
tri = []
for j in range(40):
    for i in range(40):
        n0 = j * 41 + i
        tri.append([n0, n0 + 1, n0 + 41])
        tri.append([n0 + 1, n0 + 42, n0 + 41])
f.prepare(len(x), len(tri), np.array(tri), x, y, meshtype="m", cartesian=True)
f.set_preconditioner(precond)
rng = np.random.default_rng(21)
data = rng.normal(size=int(f._n2d))
np.save(out, f.compute(2, 2 * math.pi / 200.0, data))
"""


def _solve_in_subprocess(platform, precond):
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "x.npy")
        env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
        r = subprocess.run([sys.executable, "-c", _LEG, platform, precond, out],
                           capture_output=True, text=True, env=env,
                           timeout=600)
        assert r.returncode == 0, r.stderr[-2000:]
        return np.load(out)


@pytest.mark.parametrize("precond", ["jacobi", "vcycle"])
def test_cpu_gpu_solution_parity(precond):
    cpu = _solve_in_subprocess("cpu", precond)
    gpu = _solve_in_subprocess("gpu", precond)
    np.testing.assert_allclose(gpu, cpu, rtol=1e-9, atol=1e-9)
