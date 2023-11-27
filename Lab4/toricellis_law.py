import math

import matplotlib.pyplot as plt
import numpy
import scipy
import numpy as np


def model(y, t, h, a):
    if np.isclose(y, 0):
        return numpy.inf
    # k = a * math.sqrt(2 * 9.8)
    # s = math.sqrt(h) * (2 * k) / math.pi
    return - math.sqrt(h) * h / (2 * y ** 2)


h = 10
a = 2
t_lower_bound = 1
t_upper_bound = 15
y0 = h / 2
t = np.linspace(t_lower_bound, t_upper_bound)
y = scipy.integrate.odeint(model, y0, t, args=(h, a))
inter_func = scipy.interpolate.CubicSpline(t, y)
print(inter_func.solve(0))
plt.plot(t, y)
plt.show()
