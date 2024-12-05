class SolverNotConvergedError(Exception):
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors


class VeryStupidIdeaError(Exception):
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors


def transform_attribute(self, atr: str, lmbd, fill=None):
    """
    If atribute atr exists then transform it using given Callable lmbd, otherwise it set with fill value
    """
    if hasattr(self, atr):
        setattr(self, atr, lmbd(getattr(self, atr)))
    else:
        setattr(self, atr, fill)