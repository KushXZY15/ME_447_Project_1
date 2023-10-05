import numpy as np
import random as rand
import matplotlib.pyplot as plt


# Function to Parse Knapsack Data

def open_knapsack(knapsack):
    capacity = knapsack['capacity']
    quantity = knapsack['n_items']
    values = knapsack['item_values']
    weights = knapsack['item_weights']

    return (capacity, quantity, values, weights)

# Create Initial Knapsack Population

def create_pop(pop_size, items_quant, item_prob):
    # Create Chromosomes of 0s and 1s to Select Items for initial Population
    pop_set = np.zeros((pop_size, items_quant), dtype=int)
    # makes a matrix that has number of rows = pop_size and the number of columns = quantity of items
    for population in range(pop_size):
        # goes over all populations = each row/individual in pop_set
        for chromosome in range(items_quant):
            # for each chromosome = column/item slot, item/no item is assigned in each index with a 0 or 1
            pop_set[population, chromosome] = rand.choices([0, 1], weights=[1 - item_prob, item_prob], k=1)[0]
            # weights = [probability of choosing 0 = item_prob^compliment, ...]
            # makes k-sized list (single int this case), and [chooses index of list]

    return pop_set
    # Create Chromosomes


# Calculate the Fitness of Generated Population

def fitness_calc(pop_set, val, cap, weight):
    # Matrix multiply population (pop_size x items_quant) with item values (items_quant x 1 column vector)
    fitness = pop_set @ val
    # val is value per filled item index/pop_set column
    # returns a column vector with pop_size # of rows filled with the value that each individual/row of pop_set caries
    actual_weight = pop_set @ weight
    # same process but returns weight each individual is carrying

    for population in range(len(pop_set[:, 0])):
        # indexes each individual/row of pop_set in range(0 to # of individuals/rows - 1)
        if actual_weight[population] > cap:
            # scans actual_weight vector [with population/individual index] for overweight individuals
            # If the total weight of an individual is greater than capacity, assign fitness[individual idx] = 0
            fitness[population] = 0

    return fitness

### !!! Added function to perform fitness calculation only for CMA. Since it operates 1 population at a time got rid of population based for loop. 
### !!! Left in matrix multiplications to avoid breaking something. Can try changing later if desired.

def fitness_calc_cma(pop_set, val, cap, weight):
    # Matrix multiply population (pop_size x items_quant) with item values (items_quant x 1 column vector)
    fitness = pop_set @ val
    # val is value per filled item index/pop_set column
    # returns a column vector with pop_size # of rows filled with the value that each individual/row of pop_set caries
    actual_weight = pop_set @ weight
    # same process but returns weight each individual is carrying

    if actual_weight > cap:
        # scans actual_weight vector [with population/individual index] for overweight individuals
        # If the total weight of an individual is greater than capacity, assign fitness[individual idx] = 0
        fitness = 0

    return fitness

def mating(pop_set, fitness, mates_quant):
    mate_order = np.argsort(fitness)[::-1]
    # pop_size x 1 column vector of individual indices (row # in pop_set) in order of descending fitness
    # [::-1] means descending order
    parents = pop_set[mate_order]
    # creates descending fitness-ordered array of parents from whole rows of pop_set
    parents = parents[:mates_quant, :]
    # trims the number of rows up to but excluding the "mates_quant"'th indexed row
    # leaves all the columns/chromosomes/item slots
    parents_fitness = fitness[mate_order][:mates_quant]
    # grabs the total value of what the parents are carrying in descending order up to the number of parents selected
    # parents and their fitness are related index-wise

    return parents, parents_fitness


def crossover(parents, offspring_quant, items_quant, val, cap, weight):
    offsprings = np.zeros((offspring_quant, items_quant), dtype=int)
    # creates empty matrix with # rows = offspring_quant and # columns = items_quant
    crossing_point = np.random.random_integers(0, len(parents[0, :]))
    # generates random index of crossover between 0 and the # of items/columns/chromosomes in parents

    for offspring in range(offspring_quant):
        # indexes offspring up to given # of offspring
        p1_idx = offspring % offspring_quant
        # creates index/chooses row of parents for parent 1 of current loop offspring
        p2_idx = (offspring + 1) % offspring_quant
        # does the same for parent 2, restarts at parents[row index 0] when last offspring chosen
        offsprings[offspring, :crossing_point] = parents[p1_idx, :crossing_point]
        # populates loop offspring row with items/columns/chromosomes of parent_1 up to the crossing point column index
        offsprings[offspring, crossing_point:] = parents[p2_idx, crossing_point:]
        # populates the rest of the columns of loop offspring's row with parent_2's chromosomes
        offspring_fitness = fitness_calc(offsprings, val, cap, weight)
        # returns a column vector with offspring_quant # of rows filled with values that each offspring caries
    return offsprings, offspring_fitness
    # offspring and their fitness are related index-wise


def mutation(offsprings, prob_mutation, val, cap, weight):
    for offspring in range(len(offsprings[:, 0])):
        # indexing rows in offspring/individual offspring:
        mutated_offsprings = offsprings
        # fill mutated matrix with the same items as the offspring
        mutations = np.random.uniform(0.0, 1.0, (len(offsprings[offspring, :]), 1))
        # draws sample from uniform dist from [0, 1), creates column vector the size of number of items
        for chromosome in range(len(offsprings[offspring, :])):
            # indexing each column/item/chromosome in offspring
            if mutations[chromosome] <= prob_mutation:
                # check if mutated value in mutation chromosome is <= probability of mutation
                # if true, mutate as such:
                if offsprings[offspring, chromosome] == 0:
                    # if offspring[chromosome] == 0, i.e. is empty
                    mutated_offsprings[offspring, chromosome] = 1
                    # gives the mutated offspring an item
                else:
                    mutated_offsprings[offspring, chromosome] = 0
                    # if there is an item, take away an item

    mutated_offspring_fitness = fitness_calc(offsprings, val, cap, weight)
    # sets the fitness value of mutated offspring = the fitness of the given offspring pop

    return mutated_offsprings, mutated_offspring_fitness


def environmental_selection(parents, mutated_offsprings, pop_size, parent_fitness, mutated_offspring_fitness):
    pop_set = np.vstack((parents, mutated_offsprings))
    # creates a population by stacking parents and mutated offspring row-wise (row on top of row)
    # ex: vstack(([1,2,3],[4,5,6])) = [[1,2,3],[4,5,6]]
    fitness = np.concatenate((parent_fitness, mutated_offspring_fitness))
    # creates long fitness column vector composed of parent then mutated offspring fitness
    # np.concatenate() uses existing axis if none specified, in this case, row
    # therefore fitness is also stacked row-wise
    fitness_order = np.argsort(fitness)[::-1]
    # (rows in parents + mutated offspring) x 1 column vector
    # populated with the individual/row indices (row # in pop_set) in order of descending fitness
    pop_set = pop_set[fitness_order]
    # reorders pop_set in descending fittness
    pop_set = pop_set[:pop_size, :]
    # trims the pop_set # of rows/population size to the specified pop_size
    pop_fitness = fitness[fitness_order][:pop_size]
    # grabs the total values of what the individuals are carrying in descending order up to the size of pop
    # individuals and their fitness are related index-wise

    return pop_set, pop_fitness


def knapsack_GA(pop_size, cap, weight, val, items_quant, item_prob, mates_quant, num_generations, offspring_quant,
                prob_mutation):
    pop_set = create_pop(pop_size, items_quant, item_prob)
    best_fitnesses = np.zeros(num_generations)

    for generation in range(num_generations):
        fitness = fitness_calc(pop_set, val, cap, weight)
        parents, parent_fitness = mating(pop_set, fitness, mates_quant)
        offsprings, offspring_fitness = crossover(parents, offspring_quant, items_quant, val, cap, weight)
        mutated_offsprings, mutated_offspring_fitness = mutation(offsprings, prob_mutation, val, cap, weight)
        pop_set, pop_fitness = environmental_selection(parents, mutated_offsprings, pop_size, parent_fitness,
                                                       mutated_offspring_fitness)

        max_idx = np.argmax(pop_fitness)
        best_solution = pop_set[max_idx, :]
        best_fitness = pop_fitness[max_idx]
        best_fitnesses[generation] = best_fitness
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

        
        print('Parents Info')
        print(parents)
        print(parent_fitness)

        print('Offsprings Info')
        print(offsprings)
        print(offspring_fitness)

        print('Mutated Offsprings Info')
        print(mutated_offsprings)
        print(mutated_offspring_fitness)
        
        print('Current Generation:', generation)
        print('Best Solution So Far:', best_solution)
        print('Best Fitness So Far', best_fitness)
        '''
    '''
    print('Final Generation:')
    print('Best Solution Overall:', best_solution)
    print('Best Fitness Overaall', best_fitness)
    
    plt.figure(figsize=(12, 12))
    plt.plot(best_fitnesses, '-o', lw=3, ms=20)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.show()
    
    print('Best Fitnesses from All Generations')
    print(best_fitnesses)
    '''
    return (pop_set, pop_fitness, best_solution, best_fitness, best_fitnesses)
