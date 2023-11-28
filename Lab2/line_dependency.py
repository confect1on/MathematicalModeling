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


N = 100
E1 = []
E2 = []
E3 = []
E4 = [None]
E5 = []
for i in range(10):
    E1.append(EK(0, 5, N, 1, 0, 0, 10, 1, 0)[1])
    E2.append(exactval(0, 5, N, 1, 0, 0, 10, 1, 0)[1])
    E3.append(np.max(np.abs(E1[i] - E2[i])))
    if (i != 0):
        E4.append(np.log2(E3[i - 1] / E3[i]))
    E1[i] = E1[i][-1]
    E2[i] = E2[i][-1]
    E5.append((N))
    N *= 2
# %%
df1 = pd.DataFrame({"y_{aprx}": E1,
                    "y_{exact}": E2,
                    "|y_{aprx} - y_{exact}|": E3,
                    "Delta E": E4, }, index=E5)
df1 = df1.rename_axis('N', axis=1)
print(df1)
print('\n')
df2 = pd.DataFrame({"v_{aprx}": E1,
                    "v_{exact}": E2,
                    "|v_{aprx} - v_{exact}|": E3,
                    "Delta E": E4, }, index=E5)
df2 = df2.rename_axis('N', axis=1)
print(df2)
