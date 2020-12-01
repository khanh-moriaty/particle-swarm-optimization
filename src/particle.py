
import numpy as np

class Particle:
    
    def __init__(self, particle_id, x, f, swarm, w=0.7298, c1=1.49618, c2=1.49618):
        self.id = particle_id
        self.x = [x]
        self.f = f
        self.swarm = swarm
        
        self.v = [np.zeros_like(x)]
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        self.fitness = []
        self.best = []
        self.best_fitness = []
        self.inertia = [np.zeros_like(x)]
        self.cognitive = [np.zeros_like(x)]
        self.social = [np.zeros_like(x)]
        self.update_fitness(0)
        
    def update_fitness(self, generation):
        self.fitness.append(self.f(self.x[generation]))
        self.update_best_fitness(generation)
        self.swarm.increase_evaluation_count(self, generation)
        
    def update_best_fitness(self, generation):
        # print(generation, self.fitness[generation], self.best_fitness[generation])
        if generation == 0 or self.fitness[generation] < self.best_fitness[generation-1]:
            self.best.append(self.x[generation])
            self.best_fitness.append(self.fitness[generation])
        else:
            self.best.append(self.x[generation-1])
            self.best_fitness.append(self.fitness[generation-1])
        
    def update(self, generation):
        
        inertia = self.w * self.v[generation-1]
        cognitive = self.c1 * np.random.random(self.v[generation-1].shape) * (self.best[generation-1] - self.x[generation-1])
        social = self.c2 * np.random.random(self.v[generation-1].shape) * (self.swarm.find_neighborhood_min(self.id, generation-1) - self.x[generation-1])
        
        self.inertia.append(inertia)
        self.cognitive.append(cognitive)
        self.social.append(social)
        
        # print(generation, inertia, cognitive, social, self.v[generation-1])
        
        v = inertia + cognitive + social
        
        self.v.append(v)
        assert len(self.v)-1 == generation
        x = self.f.clip_particle(self.x[generation-1] + v)
        self.x.append(x)
        
        self.update_fitness(generation)
        
        assert len(self.inertia) == len(self.cognitive) == len(self.social) == len(self.v) == len(self.x) == len(self.best) == len(self.best_fitness)
        # print(self.v[generation], self.x[generation], self.best[generation])
        
        
        # Free up memory for large problems
        if len(self.x) > 2 and len(self.x[-1]) > 2:
            self.inertia[generation-2] = None
            self.cognitive[generation-2] = None
            self.social[generation-2] = None
            self.v[generation-2] = None
            self.x[generation-2] = None
            self.best[generation-2] = None
            self.best_fitness[generation-2] = None
        
        return inertia, cognitive, social
        
        