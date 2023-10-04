# Import Packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import copy
from functools import partial

# Import Classes

from CMAES_main import CMAES 

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

current_problem = rotated_hyper_ellipsoid

#plot_problem_3d(current_problem, ((-20,-20), (20,20)))

initial_centroid = np.random.randn(2,)
cma_es = CMAES(initial_centroid,1,10,20)

solution, fitness_history = cma_es.run(current_problem)

plt.figure(figsize=(12, 12))
plt.plot(fitness_history, '-o', lw=3, ms=20)
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.show()