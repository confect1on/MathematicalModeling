import math

import numpy as np
import scipy
import matplotlib.pyplot as plt

def model(x, t, omega0, mu):
    X, vX, _x, _vx = x
    f = [
        vX,
        -omega0 ** 2 * (1 + 2 * mu) * X + mu * omega0 ** 2 * _x,
        _vx,
        -omega0 ** 2 * _x + omega0 ** 2 * X
    ]
    return f


X0 = vX0 = _x = _vx = 1
g = 9.8
l = 1
omega0 = math.sqrt(g / l)
mu = 20
T = 10
plt.subplot(121)
t = np.linspace(0, T, 100)
solution = scipy.integrate.odeint(model, [X0, vX0, _x, _vx], t, args=(omega0, mu))
plt.plot(t, solution[:, 0], 'b', label='X(t)')
plt.plot(t, solution[:, 2], 'r', label='x(t)')
plt.legend(loc='best')
plt.subplot(122)
plt.plot(t, solution[:, 1], 'b', label="X'(t)")
plt.plot(t, solution[:, 3], 'r', label="x'(t)")
plt.legend(loc='best')
plt.show()
