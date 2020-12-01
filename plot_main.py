from plotting import plot_swarm, plot_vector
from multiprocessing import Pool
from itertools import repeat
import numpy as np
import os

def process(fi, LOG_DIR, PLOT_DIR):
    print(fi)
    a = np.load(os.path.join(LOG_DIR, fi))
    fn, ext = os.path.splitext(fi)
    fi = fi.split('_')
    func_name = fi[3]
    
    out_file = os.path.join(PLOT_DIR, "{}.gif".format(fn))
    if fi[1] == 'swarm':
        plot_swarm(a, out_file, func_name)
    else:
        plot_vector(a, out_file, func_name)

def plot():
    
    LOG_DIR = "logs"
    
    dir = os.listdir(LOG_DIR)
    dir = [fi for fi in dir if fi.startswith("plotting") and os.path.splitext(fi)[-1].lower() == '.npy']
    dir.sort()
    
    PLOT_DIR = "plots"
    os.makedirs(PLOT_DIR, exist_ok=True)
    with Pool(8) as pool:
        pool.starmap(process_swarm, zip(dir, repeat(LOG_DIR), repeat(PLOT_DIR)))
    
    for fi in dir[::2]:
        process(fi, LOG_DIR, PLOT_DIR)
        
if __name__ == '__main__':
    plot()