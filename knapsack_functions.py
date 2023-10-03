import numpy as np
import random as rand
import matplotlib.pyplot as plt

# Function to Parse Knapsack Data

def open_knapsack(knapsack):
    capacity = knapsack['capacity']
    quantity = knapsack['n_items']
    values = knapsack['item_values']
    weights = knapsack['item_weights']

    return (capacity,quantity,values,weights)
'''
# OLD to Create Initial Population
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

# OLD Function to Calculate Fitness Value of Population Based on Value of Chromosomes
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

'''

### Create Initial Knapsack Population

def create_pop(pop_size,items_quant,item_prob):
    # Create Chromosomes of 0s and 1s to Select Items for initial Population
    pop_set = np.zeros((pop_size,items_quant),dtype=int)
    # makes a matrix that has number of columns = size of population and the number of rows = quantity of items
    for population in range(pop_size):
        # for each column/individual in pop_set
        for chromosome in range(items_quant):
            # for each "item slot", item/no item assigned in each index with a 0 or 1 with a specified probability
            pop_set[population,chromosome] = rand.choices([0,1],weights=[1-item_prob,item_prob],k=1)[0]
            # what is the [0] for?
    return(pop_set)
    # Create Chromosomes

### Calculate the Fitness of Generated Population

def fitness_calc(pop_set,val,cap,weight):
    # Multiply row by column Populations with Item Values to Get Fitness
    fitness = pop_set @ val
    # ! val is row vector with each item's value, dimensions of matrices are not correct for multiplication
    actual_weight = pop_set @ weight
    # ! what is shape of actual_weight? looks like returns scalar


    for population in range(len(pop_set[:,0])):
        # searches through actual_weight for overweight individuals
        if actual_weight[population] > cap:
            # Check for overweight individuals and assign fitness = 0 if overweight
            fitness[population] = 0
            # ! is fitness indexable?

    return(fitness)

def mating(pop_set,fitness,mates_quant):
    mate_order = np.argsort(fitness)[::-1]

    # ! supposed to give column indexes of pop_set sorted in order of descending fitness. syntax of [::-1]?
    parents = pop_set[mate_order]   #creates fitness-ordered array of parents
    parents = parents[:mates_quant, :]  #chooses number of parents to mate
    parents_fitness = fitness[mate_order][:mates_quant] #grabs fitness of parents in order of decreasing fitness up to the number of parents ???

    return(parents,parents_fitness)

def crossover(parents,offspring_quant,items_quant,val,cap,weight):
    offsprings = np.zeros((offspring_quant,items_quant),dtype=int)  #creates empty matrix with offspring amount number of columns and item quant rows
    crossing_point = np.random.random_integers(0,len(parents[0,:])) #generates random index of crossover between 0 and the number of items/rows in parents

    for offspring in range(offspring_quant):    #creates offspring up to the specified amount

        p1_idx = offspring%offspring_quant  #creates indice for current offspring column (filled with items_quant rows)
        p2_idx = (offspring+1)%offspring_quant  #same as before but shifted right one, restarts at 0 when p1=last

        offsprings[offspring, :crossing_point] = parents[p1_idx, :crossing_point]   #populates offsprings array
        offsprings[offspring, crossing_point:] = parents[p2_idx, crossing_point:]   # ! always overwrites cross point with parent 2's crossing_point value

        offspring_fitness = fitness_calc(offsprings,val,cap,weight)

    return(offsprings, offspring_fitness)

def mutation(offsprings,prob_mutation,val,cap,weight):
    for offspring in range(len(offsprings[:,0])):
        # for individual offspring:
        mutated_offsprings = offsprings
        # fill mutated matrix with the same items as the offspring
        mutations = np.random.uniform(0.0,1.0, (len(offsprings[offspring,:]),1))
        # draws sample from uniform dist from [0, 1), creates column vector the size of offspring number of items
        for chromosome in range(len(offsprings[offspring,:])):

            if mutations[chromosome] <= prob_mutation:
                if chromosome == 0:
                    mutated_offsprings[offspring,chromosome] = 1
                else:
                    mutated_offsprings[offspring,chromosome] = 0

    mutated_offspring_fitness = fitness_calc(offsprings,val,cap,weight)

    return(mutated_offsprings,mutated_offspring_fitness)

def environmental_selection(parents,mutated_offsprings,pop_size,parent_fitness,mutated_offspring_fitness):
    pop_set = np.vstack((parents,mutated_offsprings))
    fitness = np.concatenate((parent_fitness,mutated_offspring_fitness))
    fitness_order = np.argsort(fitness)[::-1]
    pop_set = pop_set[fitness_order]
    pop_set = pop_set[:pop_size,:]
    pop_fitness = fitness[fitness_order][:pop_size]

    return(pop_set,pop_fitness)

def knapsack_GA(pop_size,cap,weight,val,items_quant,item_prob,mates_quant,num_generations,offspring_quant,prob_mutation):
    pop_set = create_pop(pop_size,items_quant,item_prob)
    best_fitnesses = np.zeros(num_generations)

    for generation in range(num_generations):

        fitness = fitness_calc(pop_set,val,cap,weight)
        parents, parent_fitness = mating(pop_set,fitness,mates_quant)
        offsprings, offspring_fitness = crossover(parents,offspring_quant,items_quant,val,cap,weight)
        mutated_offsprings, mutated_offspring_fitness = mutation(offsprings,prob_mutation,val,cap,weight)
        pop_set, pop_fitness = environmental_selection(parents,mutated_offsprings,pop_size,parent_fitness,mutated_offspring_fitness)
        
        max_idx = np.argmax(pop_fitness)
        best_solution = pop_set[max_idx,:]
        best_fitness = pop_fitness[max_idx]
        best_fitnesses[generation] = best_fitness

        print('Parents Info')
        print(parents)
        print(parent_fitness)

        print('Offsprings Info')
        print(offsprings)
        print(offspring_fitness)

        print('Mutated Offsprings Info')
        print(mutated_offsprings)
        print(mutated_offspring_fitness)

        '''
        print('Parents Info')
        print(parents)
        print(parent_fitness)

        print('Offsprings Info')
        print(offsprings)
        print(offspring_fitness)

        print('Mutated Offsprings Info')
        print(mutated_offsprings)
        print(mutated_offspring_fitness)
        '''
        print('Current Generation:', generation)
        print('Best Solution So Far:', best_solution)
        print('Best Fitness So Far', best_fitness)


    
    print('Final Generation:')
    print('Best Solution Overall:', best_solution)
    print('Best Fitness Overaall', best_fitness)

    plt.figure(figsize=(12,12))
    plt.plot(best_fitnesses,'-o', lw=3, ms=20)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.show()

    print('Best Fitnesses from All Generations')
    print(best_fitnesses)

    return(pop_set,pop_fitness,best_solution,best_fitness,best_fitnesses)