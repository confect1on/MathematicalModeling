import numpy as np
import scipy
import matplotlib.pyplot as plt


def system(x, t, k1, k2, k3):
    x1, x2, x3 = x
    dx_dt = [-k1 * x1 + k3 * x3,
             -k2 * x2 + k1 * x1,
             -k3 * x3 + k2 * x2]
    return dx_dt


x0 = [0, 10, 10]
r = 10
v1 = 50
v2 = 25
v3 = 50
k1 = r / v1
k2 = r / v2
k3 = r / v3
t_lower_bound = 0
t_upper_bound = 100
t = np.linspace(t_lower_bound, t_upper_bound, 10)
x_res = scipy.integrate.odeint(system, x0, t, args=(k1, k2, k3))
plt.plot(t, x_res[:, 0], 'b', label='x1(t)')
plt.plot(t, x_res[:, 1], 'g', label='x2(t)')
plt.plot(t, x_res[:, 2], 'r', label='x3(t)')
plt.legend(loc='best')
second_interpolated = scipy.interpolate.CubicSpline(t, -x_res[:, 1])
extrema_second = scipy.optimize.minimize(second_interpolated, 0, bounds=[(0, t_upper_bound)])
t_max = extrema_second["x"]
x_max = scipy.interpolate.CubicSpline(t, x_res[:, 1])(t_max)
print(f"x({t_max[0]}) = {x_max[0]}")
plt.show()