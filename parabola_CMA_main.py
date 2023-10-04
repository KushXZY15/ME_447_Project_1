###!!! Created parabola code myself (i know fancy) and adapted the rest from provided

# Import Packages
import numpy as np
import matplotlib.pyplot as plt

# Import Classes

from CMAES_main import CMAES 

# Function Definitions

def parabola(x):
    return (10*x**2)

current_problem = parabola

initial_centroid = np.random.randn(1,)
cma_es = CMAES(initial_centroid,1,10,20)

solution, fitness_history = cma_es.run(current_problem)

plt.figure(figsize=(12, 12))
plt.plot(fitness_history, '-o', lw=3, ms=20)
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.show()