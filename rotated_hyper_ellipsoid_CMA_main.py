# Import Packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from functools import partial
from plot_all_fitness import plot_all_fitness
from CMAES_main import CMAES 

# Adapted from ME447 Module Modfied by Kush Patel and Ben Goddard

# Function Definitions

def range_from_bounds(bounds, resolution):
    (minx,miny),(maxx,maxy) = bounds
    x_range = np.arange(minx, maxx, (maxx-minx)/resolution)
    y_range = np.arange(miny, maxy, (maxy-miny)/resolution)
    return x_range, y_range

def plot_problem_3d(problem, bounds, ax=None, resolution=100.,
                    cmap=cm.viridis_r, rstride=10, cstride=10,
                    linewidth=0.15, alpha=0.65):
    """Plots a given benchmark problem in 3D mesh."""

    x_range, y_range = range_from_bounds(bounds, resolution=resolution)

    X, Y = np.meshgrid(x_range, y_range)
    Z = problem(X,Y)

    if not ax:
        fig = plt.figure(figsize=(11,6))
        # ax = fig.gca(projection='3d')
        ax = plt.axes(projection="3d") # For new matplotlib version

    cset = ax.plot_surface(X, Y, Z, cmap=cmap, rstride=rstride, cstride=cstride, linewidth=linewidth, alpha=alpha)
    plt.show()


def plot_problem_contour(problem, bounds, optimum=None,
                          resolution=100., cmap=cm.viridis_r,
                          alpha=0.45, ax=None):
    """Plots a given benchmark problem as a countour."""
    x_range, y_range = range_from_bounds(bounds, resolution=resolution)

    X, Y = np.meshgrid(x_range, y_range)
    Z = problem(X,Y)

    if not ax:
        fig = plt.figure(figsize=(6,6))
        ax = fig.gca()
        ax.set_aspect('equal')
        ax.autoscale(tight=True)

    cset = ax.contourf(X, Y, Z, cmap=cmap, alpha=alpha)

    if optimum:
        ax.plot(optimum[0], optimum[1], 'bx', linewidth=4, markersize=15)

def plot_cov_ellipse(pos, cov, volume=.99, ax=None, fc='lightblue', ec='darkblue', alpha=1, lw=1):
    ''' Plots an ellipse that corresponds to a bivariate normal distribution.
    Adapted from http://www.nhsilbert.net/source/2014/06/bivariate-normal-ellipse-plotting-in-python/'''
    from scipy.stats import chi2
    from matplotlib.patches import Ellipse

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    kwrg = {'facecolor':fc, 'edgecolor':ec, 'alpha':alpha, 'linewidth':lw}

    # Width and height are "full" widths, not radius
    width, height = 2 * np.sqrt(chi2.ppf(volume,2)) * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwrg)
    ax.add_artist(ellip)

def _rotated_hyper_ellipsoid_impl(x1,x2,shift_x1,shift_x2):
    return ((np.sqrt(3)/2*(x1 - shift_x1) + 1/2*(x2 - shift_x2))**2 + 5*(np.sqrt(3)/2*(x2 - shift_x2) + 1/2*(x1 - shift_x1))**2)

rotated_hyper_ellipsoid = partial(_rotated_hyper_ellipsoid_impl,shift_x1=2.0,shift_x2=2.0)

# Run Rotated Hyper Ellipsoid CMA

pop_size_set = np.array([5,10,15,20,30,40,50],dtype=int)
n_generations = 20
sigma = 2.0

all_fitness_histories = np.zeros((len(pop_size_set),n_generations))
current_combonation = 'Rotated Hyper Ellipsoid - Sigma {} '.format(sigma)

idx = 0
initial_centroid = np.random.randn(2,)
current_problem = rotated_hyper_ellipsoid

for pop_size in pop_size_set:
    cma_es = CMAES(initial_centroid,sigma,int(pop_size),n_generations)
    solution, fitness_history = cma_es.run(current_problem)
    all_fitness_histories[idx,:] = fitness_history
    idx += 1

plot_all_fitness(n_generations,pop_size_set,all_fitness_histories,current_combonation,minima=True)

#plot_problem_3d(current_problem, ((-20,-20), (20,20)))