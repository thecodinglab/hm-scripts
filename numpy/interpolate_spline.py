import numpy as np
import scipy.interpolate as ip

import matplotlib.pyplot as plt

##########################
# Definitionen           #
##########################

data = np.array([
    [8., 11.2],
    [10., 13.4],
    [12., 15.3],
    [14., 19.5],
])

bc_type = 'natural' # 'natural', 'periodic', 'not-a-knot'

##########################
# Spline Interpolation   #
##########################

x = data[:, 0]
y = data[:, 1]

spline = ip.CubicSpline(x, y, bc_type=bc_type)

##########################
# Plot                   #
##########################

x_plot = np.linspace(x[0], x[-1], 1000)
y_plot = spline(x_plot)

plt.plot(x_plot, y_plot, label='spline')
plt.plot(x, y, 'o', label='data')

plt.legend(loc='best')
plt.show()