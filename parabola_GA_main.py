from parabola_GA_functions import *

# Constants for GA
pop_size = 50

pop_spread = 1000

mates_quant = 25

children_quant = mates_quant - 1

num_generations = 500

prob_mutation = 0.05

mutation_spread = 0.05

pop_set,pop_fitness,best_solution,best_fitness,best_fitnesses = parabola_ga(pop_size, pop_spread, mates_quant, num_generations, children_quant, mutation_spread, prob_mutation)