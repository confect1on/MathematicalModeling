import matplotlib.pyplot as plt
import numpy as np
import scipy


def first_curve(x, alpha1, beta1, gamma1):
    return (alpha1 / beta1) * x - gamma1 / beta1


def second_curve(x, alpha2, beta2, gamma2):
    return (beta2 / alpha2) * x - gamma2 / alpha2


def model(z, t, alpha1, alpha2, beta1, beta2, gamma1, gamma2):
    x, y = z
    f = [
        -alpha1 * x - beta1 * y + gamma1,
        -alpha2 * y - beta2 * x + gamma2
    ]
    return f


alpha1 = 3
alpha2 = 2
beta1 = beta2 = 3
gamma1 = gamma2 = 4
x0 = 90
y0 = 50
t = np.linspace(0, 100, 100)
sol = scipy.integrate.odeint(model, [x0, y0], t, args=(alpha1, alpha2, beta1, beta2, gamma1, gamma2))
plt.plot(t, sol[:, 0], 'r', label="x(t)")
plt.plot(t, sol[:, 1], 'g', label="y(t)")
plt.legend(loc="best")
plt.show()
# phase quiver
x = np.linspace(min(sol[:, 0]), max(sol[:, 0]), 20)
y = np.linspace(min(sol[:, 1]), max(sol[:, 1]), 20)
X, Y = np.meshgrid(x, y)
u, v = np.zeros(X.shape), np.zeros(Y.shape)

NI, NJ = X.shape

for i in range(NI):
    for j in range(NJ):
        x = X[i, j]
        y = Y[i, j]
        yprime = model([x, y], 0, alpha1, alpha2, beta1, beta2, gamma1, gamma2)
        u[i,j] = yprime[0]
        v[i,j] = yprime[1]

plt.quiver(X, Y, u, v, color="g")
plt.xlabel('x')
plt.ylabel('y')
plt.show()
x = np.linspace(0, 100, 100)
plt.plot(x, first_curve(x, alpha1, beta1, gamma1), 'r', label="first curve")
plt.plot(x, second_curve(x, alpha2, beta2, gamma2), 'g', label="second curve")
plt.legend(loc="best")
plt.show()