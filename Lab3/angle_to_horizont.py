import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy import integrate as it


def track1(x0, y0, alfa, v0, g, N, t):
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    vx = np.zeros_like(t)
    vy = np.zeros_like(t)

    for i in np.arange(0, N):
        x[i] = x0 + v0 * np.cos(alfa) * t[i]
        y[i] = y0 + v0 * np.sin(alfa) * t[i] - g * (t[i] ** 2) / 2
        vx[i] = v0 * np.cos(alfa)
        vy[i] = v0 * np.sin(alfa) - g * t[i]
    return x, y, vx, vy, t


def CoordX2(t):
    a = []
    for i in t:
        if ((m / k1) * (m / k1 * g + v0 * np.sin(alfa)) * (1 - np.exp(-i * k1 / m)) - m / k1 * g * i) >= 0:
            a.append((m / k1) * v0 * np.cos(alfa) * (1 - np.exp(-i * k1 / m)))
    a = np.array(a)
    return a


def CoordY2(t):
    a = []
    for i in t:
        if ((m / k1) * (m / k1 * g + v0 * np.sin(alfa)) * (1 - np.exp(-i * k1 / m)) - m / k1 * g * i) >= 0:
            a.append((m / k1) * (m / k1 * g + v0 * np.sin(alfa)) * (1 - np.exp(-i * k1 / m)) - m / k1 * g * i)
    a = np.array(a)
    return a


def RightPart(Value, t):
    R1 = Value[2]
    R2 = Value[3]
    R3 = -k2 / m * (Value[2] ** 2 + Value[3] ** 2) * 0.5 * Value[2]
    R4 = -g - k2 / m * (Value[2] ** 2 + Value[3] ** 2) * 0.5 * Value[3]
    return R1, R2, R3, R4


alfa = np.radians(60)
R = 0.2  # радиус шара
V = (4 / 3) * np.pi * R ** 3
Rho = 2700  # плотность материала
rho = 1.29  # плотность среды
m = V * Rho  # объем шара
k1 = 6 * np.pi * 1.002 * R
v0 = 80
g = 9.8
c = 0.4  # коэффицент лобового сопротивления
S = np.pi * R ** 2
k2 = 0.5 * c * rho * S
t_ = v0 * np.sin(alfa) / g
t = np.arange(0, 2 * t_, 2 * t_ / 1000)

# 1 модель 1
res = track1(0, 0, np.radians(60), 80, 9.8, 1000, t)
plt.plot(res[0], res[1])

# 2 модель 2
plt.plot(CoordX2(t), CoordY2(t), color="r")

# 3 модель 3
Value0 = [0, 0, v0 * np.cos(alfa), v0 * np.sin(alfa)]
Result = it.odeint(RightPart, Value0, t)
Result = np.array(list(filter(lambda a: a[1] >= 0, Result)))
plt.plot(Result[:, 0], Result[:, 1])
plt.show()
