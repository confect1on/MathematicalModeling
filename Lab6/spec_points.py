import numpy as np
import matplotlib.pyplot as plt

def model_linearized(z, t):
    x, y = z
    f = [
        60*x,
        42*y
    ]
    return f


def model(z, t):
    x, y = z
    f = [
        60 * x - 3 * x ** 2 - 4 * x * y,
        42 * y - 3 * y ** 2 - 2 * x * y
    ]
    return f


x = np.linspace(0, 20, 20)
y = np.linspace(0, 20, 20)
X, Y = np.meshgrid(x, y)
u, v = np.zeros(X.shape), np.zeros(Y.shape)
ul, vl = np.zeros(X.shape), np.zeros(Y.shape)

NI, NJ = X.shape

for i in range(NI):
    for j in range(NJ):
        x = X[i, j]
        y = Y[i, j]
        yprime_l = model_linearized([x, y], 0)
        ul[i, j] = yprime_l[0]
        vl[i, j] = yprime_l[1]
        yprime = model([x, y], 0)
        u[i,j] = yprime[0]
        v[i,j] = yprime[1]

plt.quiver(X, Y, u, v, color="g")
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.quiver(X, Y, ul, vl, color="b")
plt.xlabel('x')
plt.ylabel('y')
plt.show()