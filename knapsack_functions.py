import numpy as np
import random as rand

# Function to Parse Knapsack Data

def open_knapsack(knapsack):
    capacity = knapsack['capacity']
    quantity = knapsack['n_items']
    values = knapsack['item_values']
    weights = knapsack['item_weights']

    return (capacity,quantity,values,weights)

# Function to Create Initial Population
def create_pop(pop,cap,items,quant,weight):
    pop_set = np.zeros((pop,quant),dtype=float)

    # Create Chromosomes
    for population in range(pop):
        pop_set[population,:] = rand.sample(items.tolist(),len(items))
        actual_weight = 0
        for chromosome in pop_set[population,:]:
            actual_weight += weight[int(chromosome)]
        
        # Remove Excess Chromosomes to Meet Capacity
        excess = 0
        while actual_weight > cap:
            excess -= 1
            pop_set[population,excess] = np.nan
            actual_weight = 0
            for chromosome in pop_set[population,:excess]:
                actual_weight += weight[int(chromosome)]

    return(pop_set)

# Function to Calculate Fitness Value of Population Based on Value of Chromosomes
def fitness_calc(pop_set,val):
    fitness = np.zeros(len(pop_set[:,0]))

    # Add Value of Current Chromosome Until NaN is Found
    for population in range(len(fitness)):
        for chromosome in pop_set[population,:]:
            if np.isnan(chromosome) == False:
                fitness[population] += val[int(chromosome)]
            else:                
                break

    return(fitness)