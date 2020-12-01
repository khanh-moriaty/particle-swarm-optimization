from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from functions import FunctionFactory

import numpy as np

def plot(a, out_file, func_name):
    factory = FunctionFactory.getInstance()
    f = factory.getFunction(func_name, num_var=2)
    
    domain_size = (f._upper_bound - f._lower_bound) / 2
    lo = f._optimal - domain_size
    hi = f._optimal + domain_size
    a = np.clip(a, lo + domain_size / 50, hi - domain_size / 50)
    
    xlist = np.linspace(lo[0], hi[0], 100)
    ylist = np.linspace(lo[1], hi[1], 100)
    
    X, Y = np.meshgrid(xlist, ylist)
    Z = f(np.array([X, Y]))

    fig = plt.figure()
    fig.set_tight_layout(True)
    ax = fig.add_subplot(111)
    ax.set_title(func_name.upper())
    
    cp = ax.contourf(X, Y, Z, levels=50, cmap=plt.cm.bone)
    fig.colorbar(cp)
    scatter = ax.scatter(a[0, :, 0], a[0, :, 1], c='red', s=20)
    
    def update(i):
        i = np.clip(i, 0, 50)
        label = "{} Function. Generation {}".format(func_name.capitalize(), i)
        ax.set_title(label)
        
        scatter.set_offsets(a[i])
        return ax, scatter
    
    anim = FuncAnimation(fig, update, frames=np.arange(1, 60), interval=150)
    anim.save(out_file, dpi=50, writer='imagemagick')
    