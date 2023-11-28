import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd


def EK(T0, T1, N, m, y0, v0, g, k1, k2):
    t = np.linspace(T0, T1, N)
    tau = np.abs(t[1] - t[0])
    y = np.zeros_like(t)
    v = np.zeros_like(t)

    f = lambda v: g - (k1 / m) * v - (k2 / m) * v ** 2

    y[0] = y0
    v[0] = v0

    for i in range(1, N):
        v_tilda = v[i - 1] + tau * f(v[i - 1])
        v[i] = v[i - 1] + 0.5 * tau * (f(v_tilda) + f(v[i - 1]))
        y[i] = y[i - 1] + 0.5 * tau * (v[i - 1] + v_tilda)

    return [y, v, t]


m = 2.5e-3
r = 2.54e-2
c = 0.4
S = np.pi * r ** 2
rho = 1.29
k2 = rho * S * c / 2

N = 1000
T1 = 1
res1 = EK(-0.132, T1, N, m, 0, 0, 10, 0, k2)

plt.plot(res1[-1], res1[0], label='calc')
p = (res1[0] >= 3.5).nonzero()[0][0]
t_o = res1[-1][p]
f_t_o = res1[0][p]
plt.plot(t_o, f_t_o, 'o')

exper_f = [0, 0.075, 0.260, 0.525, 0.870, 1.27, 1.73, 2.23, 2.77, 3.35]
exper_t = [-0.132, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

plt.plot(exper_t, exper_f, 'o', label='exper')
plt.legend()
plt.show()