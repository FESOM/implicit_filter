
class SolverNotConvergedError(Exception):
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors

class VeryStupidIdeaError(Exception):
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors

class TheHollyHandErrorOfAntioch(Exception):
    def __init__(self):
        message = "Then shalt thou count to two, no more, no less. Two shall be the number thou shalt filter, and the number of the filter shall be two."
        super().__init__(message)
        self.errors = ["Three shalt thou not count,"]

class SizeMissmatchError(Exception):
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors

def transform_attribute(self, atr: str, lmbd, fill=None):
    """
    If attribute atr exists, then transform it using given Callable lmbd; otherwise it set with fill value
    """
    if hasattr(self, atr):
        setattr(self, atr, lmbd(getattr(self, atr)))
    else:
        setattr(self, atr, fill)