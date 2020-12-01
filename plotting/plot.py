from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist 

from functions import FunctionFactory
from functions.rosenbrock_function import RosenbrockFunction

from itertools import combinations
from functools import partial

import numpy as np

def plot_swarm(a, out_file, func_name):
    factory = FunctionFactory.getInstance()
    f = factory.getFunction(func_name, num_var=2)
    
    domain_size = (f._upper_bound - f._lower_bound) / 2
    lo = f._optimal - domain_size
    hi = f._optimal + domain_size
    
    lo = np.full_like(f._optimal, f._lower_bound)
    hi = np.full_like(f._optimal, f._upper_bound)
    
    a = np.clip(a, lo, hi)
    
    INTERVALS = 1000
    
    xlist = np.linspace(lo[0], hi[0], INTERVALS)
    ylist = np.linspace(lo[1], hi[1], INTERVALS)
    
    X, Y = np.meshgrid(xlist, ylist)
    Z = f(np.array([X, Y]))
    Z[:, int((f._upper_bound-lo[0])*INTERVALS/(hi[0]-lo[0])):] = np.max(Z)
    Z[int((f._upper_bound-lo[1])*INTERVALS/(hi[1]-lo[1])):,] = np.max(Z)
    Z -= np.min(Z) - 1

    fig = plt.figure()
    fig.set_size_inches(15, 10, True)
    fig.set_tight_layout(True)
    fig.suptitle(func_name.upper(), size='xx-large')
    
    ax_swarm = fig.add_subplot(121, aspect='equal')
    ax_swarm.set_xlim(lo[0], hi[0])
    ax_swarm.set_ylim(lo[1], hi[1])
    ax_swarm.set_xlabel('Swarm Population', size='xx-large')
    ax_dist = fig.add_subplot(122, aspect='equal')
    ax_dist.set_xlim(-2*(hi-lo)[0], +2*(hi-lo)[0])
    ax_dist.set_ylim(-2*(hi-lo)[1], +2*(hi-lo)[1])
    ax_dist.set_xlabel('Distance between particles', size='xx-large')
    
    
    if isinstance(f, RosenbrockFunction):
        locator = ticker.LogLocator()
    else:
        locator = ticker.AutoLocator()
    
    cp = ax_swarm.contourf(X, Y, Z, levels=100, locator=locator, cmap=plt.cm.bone)
    # fig.colorbar(cp)
    scatter_swarm = ax_swarm.scatter(a[0, :, 0], a[0, :, 1], c='red', s=20)
    scatter_dist = ax_dist.scatter([], [], c='blue', s=20)
    
    def update(i):
        i = np.clip(i, 0, 50)
        distances = np.array([x[1] - x[0] for x in combinations(a[i], 2)])
        label = "{} Function\nGeneration {:02d}".format(func_name.capitalize(), i)
        fig.suptitle(label, size='xx-large')
        
        scatter_swarm.set_offsets(a[i])
        scatter_dist.set_offsets(distances)
        return fig, scatter_swarm, scatter_dist
    
    anim = FuncAnimation(fig, update, frames=np.arange(1, 60), interval=150)
    anim.save(out_file, dpi=100, writer='imagemagick')

def plot_vector(a, out_file, func_name):
    factory = FunctionFactory.getInstance()
    f = factory.getFunction(func_name, num_var=2)
    
    domain_size = (f._upper_bound - f._lower_bound) / 2
    lo = f._optimal - domain_size
    hi = f._optimal + domain_size
    
    lo = np.full_like(f._optimal, f._lower_bound)
    hi = np.full_like(f._optimal, f._upper_bound)
    
    INTERVALS = 1000
    
    xlist = np.linspace(lo[0], hi[0], INTERVALS)
    ylist = np.linspace(lo[1], hi[1], INTERVALS)
    
    X, Y = np.meshgrid(xlist, ylist)
    Z = f(np.array([X, Y]))
    Z[:, int((f._upper_bound-lo[0])*INTERVALS/(hi[0]-lo[0])):] = np.max(Z)
    Z[int((f._upper_bound-lo[1])*INTERVALS/(hi[1]-lo[1])):,] = np.max(Z)
    Z -= np.min(Z) - 1

    fig = plt.figure()
    fig.set_size_inches(12, 10, True)
    fig.suptitle(func_name.upper(), size='xx-large')
    
    ax = fig.add_subplot(111,  aspect='equal')
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    
    if isinstance(f, RosenbrockFunction):
        locator = ticker.LogLocator()
    else:
        locator = ticker.AutoLocator()
    
    cp = ax.contourf(X, Y, Z, levels=100, locator=locator, cmap=plt.cm.bone)
    # fig.colorbar(cp)
    i=0
    colors = ['black', 'green', 'blue', 'red']
    arrow_list = a[i+1, :4, :]
    arrow_names = ['Inertia', 'Cognitive', 'Social', 'Velocity']
    arrows = []
    arrows.append([ax.arrow(*a[i, 4, :], *vector, label=arrow_name, color=color, 
                            width=domain_size/100, head_width=domain_size/40, head_length=domain_size/20) 
                   for vector, color, arrow_name in zip(arrow_list, colors, arrow_names)])
    ax.legend(handles=arrows[0], bbox_to_anchor=(1.05, 1), loc='upper left')
    
    scatter = ax.scatter(*a[0, 4, :], c='red', s=200)
    
    def update(i, arrows):
        # print(i)
        i = np.clip(i, 0, 101)
        odd = i % 2
        i = i // 2
        label = "{} Function\nAdventure of The Best Particle\nGeneration {:02d}".format(func_name.capitalize(), i)
        fig.suptitle(label, size='xx-large')
        
        scatter.set_offsets(a[i, 4, :])
        for arrow in arrows[-1]: arrow.remove()
        if i+1 == len(a):
            arrow_list = []
        else:
            arrow_list = a[i+1, 3:4, :] if odd else a[i+1, :3, :]
        colors = ['red'] if odd else ['black', 'green', 'blue']
        arrows.append([ax.arrow(*a[i, 4, :], *vector, color=color, width=domain_size/100, head_width=domain_size/30, head_length=domain_size/20) for vector, color in zip(arrow_list, colors)])
        return fig, scatter, (*arrows[-1]), (*arrows[-2])
    
    anim = FuncAnimation(fig, partial(update, arrows=arrows), frames=np.arange(1, 105), interval=500)
    anim.save(out_file, dpi=100, writer='imagemagick')
    