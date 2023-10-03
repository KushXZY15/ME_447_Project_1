# Import Packages
import numpy as np
import random as rand

# Pseudo Code
# Load Data
# Create populations
# Populations will be 1xknapsack quantity array containing 0s and 1s to function as a mask to choose items
# Requires a relatively high population to ensure valid solutions are part of the 
# Import Functions
from knapsack_functions import * 

# Load Knapsack data
knapsack_A = np.load('A.npz')
knapsack_B = np.load('B.npz')

cap_A,items_quant_A,val_A,weight_A = open_knapsack(knapsack_A)
cap_B,items_quant_B,val_B,weight_B = open_knapsack(knapsack_B)

# Create Array to Hold Index Values
items_A = np.arange(0,items_quant_A)
#np.arange: returns evenly spaced values with a given interval ex: arange(0, number_of_items, step)
items_B = np.arange(0,items_quant_A)

# Constants for GA
pop_size_A = 100
pop_size_B = 100

item_prob_A = 0.01
item_prob_B = 0.2

mates_quant_A = 50
mates_quant_B = 10

offspring_quant_A = mates_quant_A - 1
offspring_quant_B = mates_quant_B - 1

crossing_point_A = int(items_quant_A/2)
crossing_point_B = int(items_quant_B/2)

num_generations_A = 1000

prob_mutation_A = 0.15

# Execute GA Algo
#pop_set_A = create_pop(pop_size_A,items_quant_A,item_prob_A)
#pop_set_B = create_pop(pop_size_B,items_quant_B,item_prob_B)

#fitness_A = fitness_calc(pop_set_A,val_A,cap_A,weight_B)
#fitness_B = fitness_calc(pop_set_B,val_B,cap_A,weight_B)

#parents_A, parents_fitness_A = mating(pop_set_A,fitness_A,mates_quant_A)
#parents_B, parents_fitness_B = mating(pop_set_B,fitness_B,mates_quant_B)

#offsprings_A, offspring_fitness_A = crossover(parents_A,crossing_point_A,offspring_quant_A,items_quant_A,val_A,cap_A,weight_A)

#parents_A2, parents_fitness_A2 = mating(offsprings_A,offspring_fitness_A,int(mates_quant_A/2))

#offsprings_A2, offspring_fitness_A2 = crossover(parents_A2,crossing_point_A,int(mates_quant_A/2)-1,items_quant_A,val_A,cap_A,weight_A)

#print(pop_set_A)
#print(fitness_A)
#print(np.sum(fitness_A))
#print(np.sum(fitness_B))

pop_set_A,pop_fitness_A,best_solution_A,best_fitness_A,best_fitnesses_A = knapsack_GA(pop_size_A,cap_A,weight_A,val_A,items_quant_A,item_prob_A,mates_quant_A,num_generations_A,offspring_quant_A,prob_mutation_A)