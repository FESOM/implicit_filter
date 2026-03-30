import pytest
from unittest.mock import patch, MagicMock

from implicit_filter.utils.utils import (
    SolverNotConvergedError,
    VeryStupidIdeaError,
    TheHollyHandErrorOfAntioch,
    SizeMissmatchError,
    transform_attribute,
    get_backend,
)

def test_errors():
    err1 = SolverNotConvergedError("msg1", ["err"])
    assert err1.errors == ["err"]
    assert str(err1) == "msg1"

    err2 = VeryStupidIdeaError("msg2", ["err"])
    assert err2.errors == ["err"]
    assert str(err2) == "msg2"

    err3 = TheHollyHandErrorOfAntioch()
    assert "Two shall be the number" in str(err3)
    assert err3.errors == ["Three shalt thou not count,"]

    err4 = SizeMissmatchError("msg4", ["err"])
    assert err4.errors == ["err"]
    assert str(err4) == "msg4"

class DummyClass:
    def __init__(self):
        self.attr1 = 10

def test_transform_attribute():
    dummy = DummyClass()
    
    # Exists, transform it
    transform_attribute(dummy, "attr1", lambda x: x * 2, fill=0)
    assert dummy.attr1 == 20
    
    # Does not exist, fill it
    transform_attribute(dummy, "attr2", lambda x: x * 2, fill=99)
    assert dummy.attr2 == 99

def test_get_backend_cpu():
    from scipy.sparse import csc_matrix, identity, diags
    
    csc, ident, diag, cg, convers, tonumpy = get_backend("cpu")
    
    assert csc is csc_matrix
    assert ident is identity
    assert diag is diags
    
    # convers is jnp.array, tonumpy is np.array
    import jax.numpy as jnp
    import numpy as np
    assert convers is jnp.array
    assert tonumpy is np.array

@patch('warnings.warn')
def test_get_backend_gpu_fallback(mock_warn):
    # If cupy is missing or no gpu device, it falls back to CPU
    csc, ident, diag, cg, convers, tonumpy = get_backend("gpu")
    
    mock_warn.assert_called()
    assert convers.__name__ == "array" # jnp.array
    assert tonumpy.__name__ == "array" # np.array

def test_get_backend_invalid():
    with pytest.raises(NotImplementedError):
        get_backend("tpu")
