import numpy as np
from functools import partial
from CMAES_main import CMAES
from knapsack_GA_functions import *
from plot_all_fitness import plot_all_fitness

# Load Knapsack data
knapsack_A = np.load('A.npz')
knapsack_B = np.load('B.npz')

cap_A,items_quant_A,val_A,weight_A = open_knapsack(knapsack_A)
cap_B,items_quant_B,val_B,weight_B = open_knapsack(knapsack_B)

x_i_limit_A = 1
x_i_limit_B = 3

# Define the Knapsack Problem and Fitness Calculation for CMA
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

'''
# Run Knapsack A

knapsack_problem_a = partial(_knapsack_problem_impl,cap=cap_A,val=val_A,weight=weight_A,x_i_limit=x_i_limit_A)

pop_size_set_A = np.array([10,20,30,40,50,60,80,100,120,150,175,200],dtype=int)
n_generations = 500
sigma = 2.0

all_fitness_histories_A = np.zeros((len(pop_size_set_A),n_generations))
current_combonation = 'Knapsack A - Sigma {} - xi_limit {}'.format(sigma,x_i_limit_A)

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
# Run Knapsack B

knapsack_problem_B = partial(_knapsack_problem_impl,cap=cap_B,val=val_B,weight=weight_B,x_i_limit=x_i_limit_B)

pop_size_set_B = np.array([10,20,30,40,50,60,80,100,120,150,175,200],dtype=int)
n_generations = 500
sigma = 0.5

all_fitness_histories_B = np.zeros((len(pop_size_set_B),n_generations))
current_combonation = 'Knapsack B - Sigma {} - xi_limit {}'.format(sigma,x_i_limit_B)

idx = 0
initial_centroid = np.random.randn(items_quant_B,)
current_problem = knapsack_problem_B

for pop_size in pop_size_set_B:
    cma_es = CMAES(initial_centroid,sigma,int(pop_size),n_generations)
    solution_2, fitness_history_2 = cma_es.run(current_problem)
    all_fitness_histories_B[idx,:] = fitness_history_2
    idx += 1

plot_all_fitness(n_generations,pop_size_set_B,all_fitness_histories_B,current_combonation,minima=True)