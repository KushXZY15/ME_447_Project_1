# Adapted from ME447 Module Modfied by Kush Patel and Ben Goddard

# Import Packages
import numpy as np
import matplotlib.pyplot as plt
from plot_all_fitness import plot_all_fitness

# Import Classes

from CMAES_main import CMAES 

# Function Definitions

def parabola(x):
    return (10*x**2)

# Set Up Variables for CMA

pop_size_set = np.array([4,5,6,7,8,9,10,12,15],dtype=int)
n_generations = 15
sigma = 2.0

all_fitness_histories = np.zeros((len(pop_size_set),n_generations))
current_combonation = 'Parabola - Sigma {} '.format(sigma)

idx = 0
initial_centroid = np.random.randn(1,)
current_problem = parabola

for pop_size in pop_size_set:
    cma_es = CMAES(initial_centroid,sigma,int(pop_size),n_generations)
    solution, fitness_history = cma_es.run(current_problem)
    all_fitness_histories[idx,:] = fitness_history
    idx += 1

plot_all_fitness(n_generations,pop_size_set,all_fitness_histories,current_combonation,minima=True)