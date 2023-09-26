# Import Packages
import numpy as np
import random as rand

# Import Functions
from knapsack_functions import open_knapsack
from knapsack_functions import create_pop
from knapsack_functions import fitness_calc

# Load Knapsack data
knapsack_A = np.load('A.npz')
knapsack_B = np.load('B.npz')

cap_A,quant_A,val_A,weight_A = open_knapsack(knapsack_A)
cap_B,quant_B,val_B,weight_B = open_knapsack(knapsack_B)

# Create Array to Hold Index Values
items_A = np.arange(0,quant_A)
items_B = np.arange(0,quant_A)

# Constants for GA
pop = 20

# Execute GA Algo
pop_set_A = create_pop(pop,cap_A,items_A,quant_A,weight_A)
pop_set_B = create_pop(pop,cap_B,items_B,quant_A,weight_B)

fitness_A = fitness_calc(pop_set_A,val_A)
fitness_B = fitness_calc(pop_set_B,val_B)

print(pop_set_A)
print(fitness_A)