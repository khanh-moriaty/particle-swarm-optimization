import numpy as np
import math

from .function import Function

class RosenbrockFunction(Function):
    
    def __init__(self, num_var):
        super().__init__(num_var)
    
    def _function(self, x):
        A=100
        x_0 = x[:-1]
        x_1 = x[1:]
        return np.sum(A*np.square(x_1 - np.square(x_0)) + np.square(1 - x_0), axis=0)
    
    @property
    def _lower_bound(self):
        return -(32)
    
    @property
    def _upper_bound(self):
        return +(32)
    
    @property
    def _optimal(self):
        return np.ones((self.num_var,))