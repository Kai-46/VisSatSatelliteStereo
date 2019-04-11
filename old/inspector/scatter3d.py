'''
==============
3D scatterplot
==============

Demonstration of a basic scatterplot in 3D.
'''

import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np



def scatter3d(xx, yy, zz, save_file):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xx, yy, zz, c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    fig.savefig(save_file)

