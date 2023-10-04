import numpy as np
from functools import partial
from CMAES_main import CMAES
from knapsack_functions import *

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
current_problem = knapsack_problem_a

initial_centroid = np.random.randn(items_quant_A,)
cma_es = CMAES(initial_centroid,2,100,200)

solution, fitness_history_A = cma_es.run(current_problem)

plt.figure(figsize=(12, 12))
plt.plot(fitness_history_A, '-o', lw=3, ms=20)
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.show()