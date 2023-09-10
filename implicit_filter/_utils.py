class SolverNotConvergedError(Exception):
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors


class VeryStupidIdeaError(Exception):
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors
