import numpy as np
from math import exp, sqrt, sin, cos, pi, e

from .function import Function

class EggHolderFunction(Function):
    
    def __init__(self, num_var):
        super().__init__(num_var)
    
    def _function(self, x):
        assert x.shape == (2,), "This function supports only 2-dimensional variables."
        x, y = x
        return -(y + 47) * sin(sqrt(abs(x/2 + (y+47)))) - x * sin(sqrt(abs(x - (y+47))))
    
    @property
    def _lower_bound(self):
        return -512
    
    @property
    def _upper_bound(self):
        return +512
    
    @property
    def _optimal(self):
        return np.array([512, 404.2319])
    
    @property
    def _optimal_value(self):
        return -959.6407