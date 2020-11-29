import numpy as np
import math

from .function import Function

class RastriginFunction(Function):
    
    def __init__(self, num_var):
        super().__init__(num_var)
    
    def _function(self, x):
        A=10
        return A*self.num_var + np.sum(np.square(x) - A * np.cos(2*math.pi * x))
    
    @property
    def _lower_bound(self):
        return -(5.12)
    
    @property
    def _upper_bound(self):
        return +(5.12)
    
    @property
    def _optimal(self):
        return np.zeros((self.num_var,))