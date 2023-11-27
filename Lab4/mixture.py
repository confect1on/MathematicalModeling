import numpy as np
import scipy
import matplotlib.pyplot as plt


def model(x, t):
    v0, r_in, r_out, c_in = 1000, 10, 10, 0
    v = v0 + (r_in - r_out) * t
    d_v = r_in - r_out
    return x * (d_v/v - r_out) + v * r_in * c_in


x0 = 20

t_min = 0
t_max = 5
t = np.linspace(t_min, t_max)
x = scipy.integrate.odeint(model, x0, t)
inter_x = scipy.interpolate.CubicSpline(t, x)
root = list(filter(lambda t_1: t_min <= t_1 <= t_max, inter_x.solve(10)[0]))
print("Root is :", root)
print("x(root) is: ", inter_x(root))
plt.plot(t, x)
plt.xlabel('time')
plt.ylabel('x(t)')
plt.show()
