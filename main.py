from pso import PSO
import itertools
import time
import numpy as np
from multiprocessing import Pool

def task1():
    
    functions = ['rastrigin', 'rosenbrock', 'eggholder', 'ackley']
    topologies = ['ring', 'star']
    
    print('task1:')
    total_time = time.time()
    
    for F, T in itertools.product(functions, topologies):
        print(F, T, 'time: ', end='')
        t = time.time()
        pso = PSO(function=F, topology=T, seed=19520624)
        pso.optimize()
        pso.print_log()
        print(time.time() - t)
    
    print('Total time: ', time.time() - total_time)
    
def _task2(F, T, N):
    a = np.zeros((10,))
    
    for i in range(10):
        t = time.time()
        pso = PSO(num_var=10, population_size=N, max_generation=10**9+7, function=F, topology=T, seed=19520624+i)
        fitness = pso.optimize()[1]
        a[i] = fitness
        pso.print_log()
        print('\t', i, F, T, N, 'time: ', time.time() - t)
        
    return (F, N, T), np.mean(a), np.std(a)
    
def task2():
    
    functions = ['rastrigin', 'rosenbrock']
    topologies = ['ring', 'star']
    population_sizes = [128, 256, 512, 1024, 2048]
    
    print('task2:')
    total_time = time.time()
    
    with Pool(10) as pool:
        res = pool.starmap(_task2, itertools.product(functions, topologies, population_sizes))
        
    res.sort(key=lambda x: x[0])
        
    with open('logs/task2.txt', 'w') as fo:
        for (F, N, T), mean, std in res:
            fo.write("{} {} {}: {} ({})\n".format(F, N, T, mean, std))
        
    
    print('Total time: ', time.time() - total_time)
    
    
    
    
if __name__ == '__main__':
    task1()