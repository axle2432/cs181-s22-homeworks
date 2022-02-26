#####################
# CS 181, Spring 2022
# Homework 1, Problem 1
# STARTER CODE
##################

import numpy as np
import matplotlib.pyplot as plt

data = [(0., 0.),
        (1., 0.5),
        (2., 1.),
        (3., 2.),
        (4., 1.),
        (6., 1.5),
        (8., 0.5)]

def compute_loss(tau):
    loss = 0
    for n, (x_n, y_n) in enumerate(data):
        f_x_n = 0
        for i, (x_i, y_i) in enumerate(data):
            if i != n:
                f_x_n += np.exp(-np.power(np.linalg.norm(x_n - x_i), 2) / tau) * y_i
        loss += np.power(y_n - f_x_n, 2)
    return loss

for tau in (0.01, 2, 100):
    print("Loss for tau = " + str(tau) + ": " + str(compute_loss(tau)))
    
def f(x, tau):
    f_x = 0
    for x_n, y_n in data:
        f_x += np.exp(-np.power(np.linalg.norm(x - x_n), 2) / tau) * y_n
    return f_x

x = np.arange(0, 12, 0.1)
y1 = [f(x_i, 0.01) for x_i in x]
y2 = [f(x_i, 2) for x_i in x]
y3 = [f(x_i, 100) for x_i in x]

plt.figure(figsize=(6, 4), dpi=200)

plt.plot(x, y1, color='red', label=r"$\tau = 0.01$")
plt.plot(x, y2, color='green', label=r"$\tau = 2$")
plt.plot(x, y3, color='blue', label=r"$\tau = 100$")

plt.xlabel(r"$x^*$")
plt.ylabel(r"$f(x^*)$")

plt.title("Kernel lengthscale comparison")
plt.legend()
plt.tight_layout()
plt.savefig("T1_P1_plot.png", facecolor="white")
plt.show