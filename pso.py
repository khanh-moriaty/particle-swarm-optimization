from src import Swarm
from functions import FunctionFactory

import numpy as np
import os

class PSO:
    
    def __init__(self, num_var=2, population_size=32, max_generation=50, function='sphere', topology='ring', seed=42):
        np.random.seed(seed)
        self.num_var = num_var
        self.population_size = population_size
        self.max_generation = max_generation
        self.function = function
        self.topology = topology
        self.seed = seed
        self.factory = FunctionFactory.getInstance()
        self.f = self.factory.getFunction(function, num_var)
        self.swarm = Swarm(population_size, self.f, topology=topology)
        
    def optimize(self):
        swarm = self.swarm
        for G in range(self.max_generation):
            swarm.update()
            if swarm.evaluations >= 10**6: break
            
        return swarm.best[-1], swarm.best_fitness[-1]
    
    def print_log(self):
        
        function = self.function
        num_var = self.num_var
        topology = self.topology
        population_size = self.population_size
        seed = self.seed
        swarm = self.swarm
        
        os.makedirs('logs', exist_ok=True)
        
        def pprint(a):
            return ' '.join(map('{:.6f}'.format, a.astype(float)))
        
        log_file = "logs/{:02d}_{}_{}_{}_{}.txt".format(num_var, function, topology, population_size, seed)
        with open(log_file, 'w') as fo:
            fo.write("Log file for {} problem with {} variable\n".format(function, num_var))
            fo.write("Topology used: {}\n".format(topology))
            fo.write("Population size: {}\n\n".format(population_size))
            fo.write("Random seed: {}\n".format(seed))
            fo.write("Solution: {}\nBest fitness value: {:.6f}\n".format(pprint(swarm.best[-1]), swarm.best_fitness[-1]))
            fo.write("Optimal fitness value: {:.6f}\n".format(swarm.f._optimal_value))
            fo.write("Absolute different: {:.6f}\n".format(abs(swarm.f._optimal_value - swarm.best_fitness[-1])))
            
        if num_var == 2:
            
            particle = min(swarm.particles, key=lambda x: x.fitness[-1])
            a = np.array([particle.inertia, particle.cognitive, particle.social, particle.v, particle.x])
            a = a.transpose((1, 0, 2))
            print(a.shape)
            
            log_file = "logs/plotting_vector_{:02d}_{}_{}_{}_{}.npy".format(num_var, function, topology, population_size, seed)
            
            np.save(log_file, a)
                
            
            log_file = "logs/plotting_swarm_{:02d}_{}_{}_{}_{}.npy".format(num_var, function, topology, population_size, seed)
            a = np.array([particle.x for particle in swarm.particles])
            a = a.transpose((1, 0, 2))
            print(a.shape)
            np.save(log_file, a)
                

if __name__ == '__main__':
        
    seed = 19520624
        
    pso = PSO(num_var=2, population_size=32, max_generation=50, function='rastrigin', topology='star', seed=seed)
    
    best, fitness = pso.optimize()
    pso.print_log()
    print(best)
    print(fitness)