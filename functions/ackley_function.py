import numpy as np
from math import pi, e
from numpy import exp, sqrt, sin, cos

from .function import Function

class AckleyFunction(Function):
    
    def __init__(self, num_var):
        super().__init__(num_var)
    
    def _function(self, x):
        assert x.shape[0] == 2, "This function supports only 2-dimensional variables."
        x, y = x
        return -20 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) \
            - exp(0.5 * (cos(2*pi * x) + cos(2*pi * y))) + e + 20
    
    @property
    def _lower_bound(self):
        return -5
    
    @property
    def _upper_bound(self):
        return +5
    
    @property
    def _optimal(self):
        return np.zeros((self.num_var,))