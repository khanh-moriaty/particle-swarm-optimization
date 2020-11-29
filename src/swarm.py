
from .particle import Particle
import numpy as np

class Swarm:
    
    def __init__(self, population_size, f, topology='ring'):
        self.population_size = population_size
        self.f = f
        self.topology = topology
        self.evaluations = 0
        self.particles = []
        
        self.generation = 0
        self.best = []
        self.best_fitness = []
        self.best_neighbor = []
        self.init_swarm()
        
    def init_swarm(self):
        for particle_id in range(self.population_size):
            x = self.f.generate_particle()
            particle = Particle(particle_id, x, self.f, self)
            self.particles.append(particle)
            
    def update(self):
        self.generation += 1
        for particle in self.particles:
            particle.update(self.generation)
            
        # Free up memory for large problems
        if len(self.best) > 2 and len(self.best[-1]) > 2:
            self.best[self.generation-2] = None
            self.best_fitness[self.generation-2] = None
        
    def find_neighborhood_min(self, particle_id, generation):
        if self.topology == 'ring':
            best_particle = self._find_ring_neighborhood_min(particle_id, generation)
        if self.topology == 'star':
            best_particle = self._find_star_neighborhood_min(generation)
        assert best_particle, "Neighborhood topology \"{}\" is not supported".format(self.topology)
        return best_particle.best[generation]
        
    def _find_ring_neighborhood_min(self, particle_id, generation):
        left_id = (particle_id-1) % self.population_size
        right_id = (particle_id+1) % self.population_size
        
        left = self.particles[left_id]
        mid = self.particles[particle_id]
        right = self.particles[right_id]
        
        return min(left, mid, right, key=lambda x: x.best_fitness[generation])
        
    def _find_star_neighborhood_min(self, generation):
        # return min(self.particles, key=lambda x: x.best_fitness[generation])
        if len(self.best_neighbor) == generation:
            self.best_neighbor.append(min(self.particles, key=lambda x: x.best_fitness[generation]))
        return self.best_neighbor[generation]
        
    def increase_evaluation_count(self, particle, generation):
        self.evaluations += 1
        
        if len(self.best) == generation:
            self.best.append(None)
            self.best_fitness.append(np.inf)
            
        if particle.fitness[generation] < self.best_fitness[generation]:
            self.best[generation] = particle.x[generation]
            self.best_fitness[generation] = particle.fitness[generation]