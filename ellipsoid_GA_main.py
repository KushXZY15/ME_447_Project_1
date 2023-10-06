from ellipsoid_GA_functions import *
from plot_all_fitness import plot_all_fitness

# Kush Patel and Ben Goddard

# Constants for GA
pop_size = 50

pop_size_set = np.array([10,20,30,40,50,60,70,80,90,100])

pop_spread = 100

mates_quant = 25

children_quant = mates_quant - 1

num_generations = 500

prob_mutation = 0.1

mutation_spread = 0.25

all_fitness_histories = np.zeros((len(pop_size_set),num_generations))

current_combonation = 'Rotated Hyper Ellipsoid  Mutation_Prob {} - Mutation_Spread {} - Pop_Spread {}'.format(prob_mutation, mutation_spread, pop_spread)

idx = 0
for pop_size in pop_size_set:
    mates_quant = int(pop_size/2)
    children_quant = mates_quant - 1
    pop_set,pop_fitness,best_solution,best_fitness,best_fitnesses = ellipsoid_ga(pop_size, pop_spread, mates_quant, num_generations, children_quant, mutation_spread, prob_mutation)
    all_fitness_histories[idx,:] = best_fitnesses
    idx += 1

plot_all_fitness(num_generations,pop_size_set,all_fitness_histories,current_combonation,minima=False)
