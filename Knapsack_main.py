# Import Packages
import numpy as np
import random as rand
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from plot_all_fitness import plot_all_fitness
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

pop_size_A_set = np.array([10,20,30,40,50,70,80,90,100,150,200])

pop_size_A = 50
pop_size_B = 50

item_prob_A = 0.10
item_prob_B = 0.10

mates_quant_A = 25
mates_quant_B = 25

offspring_quant_A = mates_quant_A - 1
offspring_quant_B = mates_quant_B - 1

crossing_point_A = int(items_quant_A/2)
crossing_point_B = int(items_quant_B/2)

num_generations_A = 500
num_generations_B = 500

prob_mutation_A = 0.05
prob_mutation_B = 0.05

# Execute GA Algo
#pop_set_A = create_pop(pop_size_A,items_quant_A,item_p0b_B)

#fitness_A = fitness_calc(pop_set_A,val_A,cap_A,weight_B)0
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

all_fitness_histories_A = np.zeros((len(pop_size_A_set),num_generations_A))
current_combonation = 'Knapsack_A - Item_Prob {} - Mutation_Prob {}'.format(item_prob_A,prob_mutation_A)

idx = 0
for pop_size_A in pop_size_A_set:
    mates_quant_A = int(pop_size_A/2)
    offspring_quant_A = mates_quant_A - 1
    pop_set_A,pop_fitness_A,best_solution_A,best_fitness_A,best_fitnesses_A = knapsack_GA(pop_size_A,cap_A,weight_A,val_A,items_quant_A,item_prob_A,mates_quant_A,num_generations_A,offspring_quant_A,prob_mutation_A)
    all_fitness_histories_A[idx,:] = best_fitnesses_A
    idx += 1

plot_all_fitness(num_generations_A,pop_size_A_set,all_fitness_histories_A,current_combonation)

'''
# Create 2D Plot of Fitness Varying with Population Size and Generations
fig = plt.figure(figsize=(20,10))
fig.suptitle('Fitness History from {}'.format(current_combonation))

a2d = plt.subplot(121)
idx = 0
for pop_size in pop_size_A_set:
    a2d.plot(all_fitness_histories_A[idx,:],label='{}'.format(pop_size))
    idx += 1
a2d.legend()
a2d.set_xlabel('Generation Number')
a2d.set_ylabel('Best Fitness Value In Generation')

# Plot 3D Plot of Fitness Varying with Population Size and Generations

x,y = np.meshgrid(range(num_generations_A),pop_size_A_set)
z = all_fitness_histories_A

a3d = plt.subplot(122,projection='3d')
colors_list = list(range(len(y)))
colors_list = ['C{}'.format(idx) for idx in list(range(len(y)))]

a3d.plot_wireframe(x, y, z, rstride=1, cstride=0, colors=colors_list)
a3d.view_init(elev=20, azim=-135)
a3d.set_xlabel('Generation Number')
a3d.set_ylabel('Population Size')
a3d.set_zlabel('Best Fitness Value in Generation')
plt.show()
'''

np.savetxt('All Fitness Histories of {}'.format(current_combonation),all_fitness_histories_A,delimiter=',')
np.savetxt('Pop Size Set of {}'.format(current_combonation),pop_size_A_set,delimiter=',')
#np.savetxt('N-Generations of {}'.format(current_combonation),num_generations_A)

#pop_set_B,pop_fitness_B,best_solution_B,best_fitness_B,best_fitnesses_B = knapsack_GA(pop_size_B,cap_B,weight_B,val_B,items_quant_B,item_prob_B,mates_quant_B,num_generations_B,offspring_quant_B,prob_mutation_B)