# Import Packages
import numpy as np
from plot_all_fitness import plot_all_fitness
from knapsack_functions import * 

# Kush Patel and Ben Goddard

# Pseudo Code
# Load Data
# Create populations
# Populations will be 1xknapsack quantity array containing 0s and 1s to function as a mask to choose items
# Requires a relatively high population to ensure valid solutions are part of the 
# Import Functions

# Load Knapsack data
knapsack_A = np.load('A.npz')
knapsack_B = np.load('B.npz')

cap_A,items_quant_A,val_A,weight_A = open_knapsack(knapsack_A)
cap_B,items_quant_B,val_B,weight_B = open_knapsack(knapsack_B)

# Constants for GA

pop_size_A_set = np.array([10,20,30,40,50,70,80,90,100,150,200])
pop_size_B_set = np.array([10,20,30,40,50,70,80,90,100,150,200])

item_prob_A = 0.10
item_prob_B = 0.05

num_generations_A = 500
num_generations_B = 500

prob_mutation_A = 0.10
prob_mutation_B = 0.10

'''
# Run Knapsack A
all_fitness_histories_A = np.zeros((len(pop_size_A_set),num_generations_A))
current_combonation = 'Knapsack_A - Item_Prob {} - Mutation_Prob {}'.format(item_prob_A,prob_mutation_A)

idx = 0
for pop_size_A in pop_size_A_set:
    mates_quant_A = int(pop_size_A/2)
    offspring_quant_A = mates_quant_A - 1
    pop_set_A,pop_fitness_A,best_solution_A,best_fitness_A,best_fitnesses_A = knapsack_GA(pop_size_A,cap_A,weight_A,val_A,items_quant_A,item_prob_A,mates_quant_A,num_generations_A,offspring_quant_A,prob_mutation_A)
    all_fitness_histories_A[idx,:] = best_fitnesses_A
    idx += 1

plot_all_fitness(num_generations_A,pop_size_A_set,all_fitness_histories_A,current_combonation,minima=False)
'''
# Run Knapsack B
all_fitness_histories_B = np.zeros((len(pop_size_B_set),num_generations_B))
current_combonation = 'Knapsack_B - Item_Prob {} - Mutation_Prob {}'.format(item_prob_B,prob_mutation_B)

idx = 0
for pop_size_B in pop_size_B_set:
    mates_quant_B = int(pop_size_B/2)
    offspring_quant_B = mates_quant_B - 1
    pop_set_B,pop_fitness_B,best_solution_B,best_fitness_B,best_fitnesses_B = knapsack_GA(pop_size_B,cap_B,weight_B,val_B,items_quant_B,item_prob_B,mates_quant_B,num_generations_B,offspring_quant_B,prob_mutation_B)
    all_fitness_histories_B[idx,:] = best_fitnesses_B
    idx += 1

plot_all_fitness(num_generations_B,pop_size_B_set,all_fitness_histories_B,current_combonation,minima=False)

# Data Save
#np.savetxt('All Fitness Histories of {}'.format(current_combonation),all_fitness_histories_B,delimiter=',')
#np.savetxt('Pop Size Set of {}'.format(current_combonation),pop_size_B_set,delimiter=',')
#np.savetxt('N-Generations of {}'.format(current_combonation),num_generations_A)