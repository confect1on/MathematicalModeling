import math

import numpy as np
import scipy
import matplotlib.pyplot as plt

def a(t, a0, a1, b1, omega):
    return a0 + a1 * np.cos(omega * t) + b1 * np.cos(omega * t)

def model(x, t, a0, a1, b1, omega, k):
    return -k * (x - a(t, a0, a1, b1, omega))


omega = math.pi / 12
a0 = a1 = b1 = k = 1
u0 = 10
t = np.linspace(0, 100, 100)
sol = scipy.integrate.odeint(model, u0, t, args=(a0, a1, b1, omega, k))
plt.plot(t, sol, 'b', label='u(t)')
plt.plot(t, a(t, a0, a1, b1, omega), 'r', label='A(t)')
plt.legend(loc='best')
plt.show()