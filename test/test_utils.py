import pytest
from unittest.mock import patch, MagicMock

from implicit_filter.utils.utils import (
    SolverNotConvergedError,
    VeryStupidIdeaError,
    TheHollyHandErrorOfAntioch,
    SizeMissmatchError,
    transform_attribute,
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


