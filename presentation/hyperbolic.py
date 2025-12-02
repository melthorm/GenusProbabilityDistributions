import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Hyperbolic plane as upper sheet of the two-sheeted hyperboloid:
#   x^2 + y^2 - z^2 = -1,  z > 0

def hyperbolic_sheet(rmax=2.0, n=400):
    u = np.linspace(0, rmax, n)
    v = np.linspace(0, 2*np.pi, n)
    U, V = np.meshgrid(u, v)

    X = np.sinh(U) * np.cos(V)
    Y = np.sinh(U) * np.sin(V)
    Z = np.cosh(U)
    return X, Y, Z

X, Y, Z = hyperbolic_sheet()

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False, alpha = 0.5)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Hyperbolic plane: upper sheet of two-sheeted hyperboloid')

plt.show()




