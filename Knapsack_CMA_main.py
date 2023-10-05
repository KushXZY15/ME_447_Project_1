import numpy as np
from functools import partial
from CMAES_main import CMAES
from knapsack_functions import *
from plot_all_fitness import plot_all_fitness
# Load Knapsack data
knapsack_A = np.load('A.npz')
knapsack_B = np.load('B.npz')

cap_A,items_quant_A,val_A,weight_A = open_knapsack(knapsack_A)
cap_B,items_quant_B,val_B,weight_B = open_knapsack(knapsack_B)

x_i_limit_A = 3
x_i_limit_B = 1

###!!! Function that defines the knapsack problem. Takes in x_i as *args and "masks" to convert floats into 1s and 0s based on desired threshhold limit. Based on TA suggestion in discord.

def _knapsack_problem_impl(*args,cap,val,weight,x_i_limit):
    pop = np.zeros(len(args))
    idx = 0
    for item in args:
        if item > x_i_limit:
            pop[idx] = 1
        else:
            pop[idx] = 0
        idx += 1
    return (-fitness_calc_cma(pop,val,cap,weight))

knapsack_problem_a = partial(_knapsack_problem_impl,cap=cap_A,val=val_A,weight=weight_A,x_i_limit=x_i_limit_A)

pop_size_set_A = np.array([10,20,30,40,50,60,80,100,120,150,175,200],dtype=int)
n_generations = 300
sigma = 2

all_fitness_histories_A = np.zeros((len(pop_size_set_A),n_generations))
current_combonation = 'Knapsack A - Sigma {} '.format(sigma)

idx = 0
initial_centroid = np.random.randn(items_quant_A,)
current_problem = knapsack_problem_a

for pop_size in pop_size_set_A:
    cma_es = CMAES(initial_centroid,sigma,int(pop_size),n_generations)
    solution_2, fitness_history_2 = cma_es.run(current_problem)
    all_fitness_histories_A[idx,:] = fitness_history_2
    idx += 1

plot_all_fitness(n_generations,pop_size_set_A,all_fitness_histories_A,current_combonation,minima=True)

'''
plt.figure(figsize=(12, 12))
plt.plot(fitness_history_A, '-o', lw=3, ms=20)
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.show()
'''
