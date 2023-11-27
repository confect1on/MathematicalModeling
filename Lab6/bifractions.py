import matplotlib.pyplot as plt
import numpy as np

def logistic(k, x):
    return x + k * x ** 3

iterations = 1000
last = 100
n = 10000
x = 1e-5 * np.ones(n)
k = np.linspace(-5, 4.0, n)
fig, ax1 = plt.subplots(1, 1, figsize=(8, 9),
                               sharex=True)

for i in range(iterations):
    x = logistic(k, x)
    # We display the bifurcation diagram.
    if i >= (iterations - last):
        ax1.plot(k, x, ',k', alpha=.25)
ax1.set_xlim(-5, 4)
ax1.set_title("Bifurcation diagram")
plt.show()