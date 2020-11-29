import functools
import numpy as np
import math

class Function:

    def __init__(self, num_var):
        self.num_var = num_var
        
    def __call__(self, x):
        return self._function(x)
    
    def generate_particle(self):
        return np.random.uniform(self._lower_bound, self._upper_bound, (self.num_var, ))
    
    def clip_particle(self, x):
        return np.clip(x, self._lower_bound, self._upper_bound)
        
    def _function(self, x):
        return 0
    
    @property
    def _lower_bound(self):
        return 0
    
    @property
    def _upper_bound(self):
        return 0
    
    @property
    def _optimal(self):
        return 0
    
    @property
    def _optimal_value(self):
        return 0
        
    def _rastrigin_function(x, n=2, A=10):
        return A*n + np.sum(np.square(x) - A * np.cos(2*math.pi * x))
        