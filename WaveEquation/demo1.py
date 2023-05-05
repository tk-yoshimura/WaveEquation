import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from wave_equation_solver import WaveEquationSolver

# grid parameter
N = 129
L = 16
grid_interval = L / (N - 1)

# situation parameter
c = 4
mu = 0.25

# time parameter
dt = 1 / 64

# make grid
x = np.linspace(0, L, N, endpoint=True) - L / 2
y = np.linspace(0, L, N, endpoint=True) - L / 2
X, Y = np.meshgrid(x, y)

# initial state
z0 = -np.exp(-(X**2 + Y**2) * 2) * 1.25;

# solve
solver = WaveEquationSolver(N, grid_interval, z0, c, mu, dt)
zs, ts = solver.batch_update(256)

# plot
os.makedirs("../results/", exist_ok=True)

plt.close()
fig = plt.figure(figsize = (9, 8), facecolor = "white")
ax = fig.add_subplot(111, projection='3d')

print("plotting...", end='')

def plot(z):
    print(".", end='')

    ax.clear()

    ax.plot_surface(X, Y, z, cmap = plt.cm.RdBu_r, vmin=-0.125, vmax=0.125)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(-1.5, 1.5)
    plt.tight_layout()

anim = animation.FuncAnimation(fig, plot, frames = zs, interval=50)
anim.save("../results/demo1.gif")