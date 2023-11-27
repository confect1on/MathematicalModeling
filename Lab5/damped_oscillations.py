import math

import numpy as np
import scipy
import matplotlib.pyplot as plt

def model_orig(x, t, zeta, omega):
    _x, v = x
    f = [
        v,
        -2 * zeta * omega * v - omega ** 2 * _x
    ]
    return f


def model_clear(x, t, omega):
    _x, v = x
    f = [
        v,
        -omega ** 2 * _x
    ]
    return f


k = 125
c = 10
m = 1
x0 = 6
v0 = 50
omega = math.sqrt(k / m)
zeta = c / (2 * math.sqrt(k * m))
print(f"Zeta: {zeta}")
if zeta > 1:
    print("Aperiodic")
elif np.isclose(zeta, 1):
    print("Aperiodic limit")
else:
    print("Weak attenuation")
t = np.linspace(0, 10, 100)
sol_orig = scipy.integrate.odeint(model_orig, [x0, v0], t, args=(zeta, omega))
sol_clear = scipy.integrate.odeint(model_clear, [x0, v0], t, args=(omega,))
amplitude = max(abs(max(sol_clear[:, 0])), abs(min(sol_clear[:, 0])))
print(f"amplitude: {amplitude}")
phase_angle = math.acos(x0 / amplitude)
print(f"phase angle: {phase_angle}")
temporary_delay = phase_angle / omega
print(f"circular frequency: {omega}")
print(f"temporary delay: {temporary_delay}")
plt.subplot(221)
plt.plot(t, sol_orig[:, 0], 'b', label='x(t)')
plt.legend(loc='best')
plt.subplot(222)
plt.plot(t, sol_orig[:, 1], 'b', label="x'(t)")
plt.legend(loc='best')
plt.subplot(223)
plt.plot(t, sol_clear[:, 0], 'b', label='u(t)')
plt.legend(loc='best')
plt.subplot(224)
plt.plot(t, sol_clear[:, 1], 'b', label="u'(t)")
plt.legend(loc='best')
plt.show()