import numpy as np

from .function import Function

class SphereFunction(Function):
    
    def __init__(self, num_var):
        super().__init__(num_var)
    
    def _function(self, x):
        return np.sum(np.square(x))
    
    @property
    def _lower_bound(self):
        return -(10**9+7)
    
    @property
    def _upper_bound(self):
        return +(10**9+7)
    
    @property
    def _optimal(self):
        return np.zeros((self.num_var,))