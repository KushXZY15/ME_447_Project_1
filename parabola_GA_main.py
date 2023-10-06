from parabola_GA_functions import *
from plot_all_fitness import plot_all_fitness

# Kush Patel and Ben Goddard

# Constants for GA
pop_size_set = np.array([6,10,25,50,75,100])

pop_spread = 1000000

mates_quant = 25

children_quant = mates_quant - 1

num_generations = 50

prob_mutation = 0.5

mutation_spread = 0.5

all_fitness_histories = np.zeros((len(pop_size_set),num_generations))

current_combonation = 'Parabola  Mutation_Prob {} - Mutation_Spread {} - Pop_Spread {}'.format(prob_mutation, mutation_spread, pop_spread)

idx = 0
for pop_size in pop_size_set:
    mates_quant = int(pop_size/2)
    children_quant = mates_quant - 1
    pop_set,pop_fitness,best_solution,best_fitness,best_fitnesses = parabola_ga(pop_size, pop_spread, mates_quant, num_generations, children_quant, mutation_spread, prob_mutation)
    all_fitness_histories[idx,:] = best_fitnesses
    idx += 1

plot_all_fitness(num_generations,pop_size_set,all_fitness_histories,current_combonation,minima=False)