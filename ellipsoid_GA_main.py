from ellipsoid_GA_functions import *


# Constants for GA
pop_size = 50

pop_spread = 100

mates_quant = 25

children_quant = mates_quant - 1

num_generations = 1000

prob_mutation = 0.1

mutation_spread = 0.25

pop_set,pop_fitness,best_solution,best_fitness,best_fitnesses = ellipsoid_ga(pop_size, pop_spread, mates_quant, num_generations, children_quant, mutation_spread, prob_mutation)