import numpy as np
from .variable import *


class MSE:
    def __call__(self, expected:Variable, real:Variable):
        return average(square(expected - real))
