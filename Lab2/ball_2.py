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


r = 0.17
mu = 3870
rho = 1.26e3
c = 0.4
S = np.pi * r ** 2

k1 = 6 * np.pi * r * mu
k2 = rho * S * c / 2

N = 1000
T1 = 2
res2 = EK(0, T1, N, 100, 0, 0, 10, k1, k2)

plt.plot(res2[-1], res2[0])
plt.title("Changes of height throwing")
plt.ylabel("$y(t),м$")
plt.xlabel("$t,c$")
plt.show()

plt.plot(res2[-1], res2[1])
plt.title("Changes of velocity throwing")
plt.ylabel("$v(t),\\frac{м}{с}$")
plt.xlabel("$t,c$")
plt.show()