import numpy as np
import matplotlib.pyplot as plt

plt.title("Transformed training data and optimal decision boundary")
plt.xlabel(r'$\phi_1(x) = x$')
plt.ylabel(r'$\phi_2(x) = -8/3x^2 + 2/3x^4$')

x = [-3, -2, -1, 0, 1, 2, 3]
phi_1 = [-8/3 * x**2 + 2/3 * x**4 for x in x]
y = [1, 1, 0, 1, 0, 1, 1]

print(phi_1)

colormap = np.array(['tab:red', 'tab:blue'])

plt.scatter(x, phi_1, c=colormap[y])
plt.axhline(y = -1, c='black', linestyle=':')

plt.savefig("svm.png")
plt.show()

plt.cla()

plt.title("Alternate feature mapping")
plt.xlabel(r'$\phi\prime_1(x) = x$')
plt.ylabel(r'$\phi\prime_2(x) = -31/12x^2 + 7/12x^4$')

phi_prime_1 = [-31/12 * x**2 + 7/12 * x**4 for x in x]

print(phi_prime_1)

plt.scatter(x, phi_prime_1, c=colormap[y])
plt.axhline(y = -1.5, c='black', linestyle=':')

plt.savefig("svm_alt.png")
plt.show()

