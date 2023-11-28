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


R = 2.7
m = 80
rho = 1.29
c = 1.33
S = (np.pi * R ** 2) / 2

k2 = rho * S * c / 2

N = 1000
T1 = 10
res1 = EK(0, T1, N, m, 0, 0, 10, 0, k2)

p = (np.diff(res1[1]) <= 0.001).nonzero()[0][0]
t_o = res1[-1][p]
f_t_o = res1[1][p]

print(f"In point {t_o:.2f} velocity becomes persistently {f_t_o:.1f}")
# %%
plt.plot(res1[-1], res1[1])
plt.ylabel("$v(t),\\frac{м}{с}$")
plt.xlabel("$t,c$")
plt.plot(t_o, f_t_o, 'o')
plt.show()