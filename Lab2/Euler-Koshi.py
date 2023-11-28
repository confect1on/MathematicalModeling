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


def exactval(T0, T1, N, m, y0, v0, g, k1, k2):
    t = np.linspace(T0, T1, N)
    y = np.zeros_like(t)
    v = np.zeros_like(t)
    y[0] = y0
    v[0] = v0
    for i in range(1, N):
        y[i] = (g * m / k1) * (t[i] + m * np.exp(-k1 * t[i] / m) / k1) - g * m ** 2 / k1 ** 2
        v[i] = (g * m / k1) * (1 - np.exp(-k1 * t[i] / m))

    return [y, v, t]


res = EK(0, 5, 100, 1, 0, 0, 10, 2, 0)
ex_res = exactval(0, 5, 100, 1, 0, 0, 10, 2, 0)

df = pd.DataFrame({"y(t)": res[0], "v(t)": res[1], "$|y-y_1|$": np.max(np.abs(res[0] - ex_res[0])),
                   "$$|v-v_1|$$": np.max(np.abs(res[1] - ex_res[1]))})

print(df)
plt.plot(res[-1], res[1], '-o')
plt.plot(ex_res[-1], ex_res[1])
plt.show()
