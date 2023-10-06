import numpy as np
import matplotlib.pyplot as plt

# Kush Patel and Ben Goddard

def plot_all_fitness(num_generations,pop_size_set,all_fitness_history,current_combonation,minima):
    # Create 2D Plot of Fitness Varying with Population Size and Generations
    fig = plt.figure(figsize=(20,10))
    fig.suptitle('Fitness History from {}'.format(current_combonation))

    a2d = plt.subplot(121)
    idx = 0
    for pop_size in pop_size_set:
        a2d.plot(all_fitness_history[idx,:],label='{}'.format(pop_size))
        idx += 1
    a2d.legend()
    a2d.set_xlabel('Generation Number')
    a2d.set_ylabel('Best Fitness Value In Generation')

    x,y = np.meshgrid(range(num_generations),pop_size_set)
    z = all_fitness_history

    a3d = plt.subplot(122,projection='3d')
    colors_list = list(range(len(y)))
    colors_list = ['C{}'.format(idx) for idx in list(range(len(y)))]

    a3d.plot_wireframe(x, y, z, rstride=1, cstride=0, colors=colors_list)
    if minima == True:       
        a3d.view_init(elev=20, azim=45)
    else:
        a3d.view_init(elev=20, azim=-135)
    a3d.set_xlabel('Generation Number')
    a3d.set_ylabel('Population Size')
    a3d.set_zlabel('Best Fitness Value in Generation')
    plt.show()
