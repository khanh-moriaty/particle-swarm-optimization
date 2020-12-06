import numpy as np
from math import pi, e
from numpy import exp, sqrt, sin, cos

from .function import Function

class EasomFunction(Function):
    
    def __init__(self, num_var):
        super().__init__(num_var)
    
    def _function(self, x):
        assert x.shape[0] == 2, "This function supports only 2-dimensional variables."
        x, y = x
        return -cos(x) * cos(y) * exp(-((x - pi)**2 + (y - pi)**2))
    
    @property
    def _lower_bound(self):
        return -100
    
    @property
    def _upper_bound(self):
        return +100
    
    @property
    def _optimal(self):
        return np.array([pi, pi])
    
    @property
    def _optimal_value(self):
        return -1