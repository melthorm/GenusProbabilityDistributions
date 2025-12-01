import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting

def plot_hyperboloid_and_disk(points_hyperboloid, points_disk, vertices=None, c=None):
    """
    Plots hyperboloid points and their corresponding mapped points on the unit disk.
    
    Args:
        points_hyperboloid: list or array of [N,3] points on the hyperboloid
        points_disk: list or array of [N,2] points on the unit disk
        vertices: optional [2n,3] hyperboloid polygon vertices
        c: optional center point on hyperboloid
    """
    points_hyperboloid = np.array(points_hyperboloid)
    points_disk = np.array(points_disk)
    
    fig = plt.figure(figsize=(12,6))
    
    # --- Hyperboloid plot ---
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(points_hyperboloid[:,0], points_hyperboloid[:,1], points_hyperboloid[:,2],
             'bo', label='Random interior points')
    if vertices is not None:
        vertices = np.array(vertices)
        ax1.plot(vertices[:,0], vertices[:,1], vertices[:,2], 'r-o', label='Polygon vertices')
    if c is not None:
        ax1.scatter([c[0]], [c[1]], [c[2]], color='red', s=50, label='Center')
    ax1.set_title("Hyperboloid Polygon + Points")
    ax1.set_xlabel("X0")
    ax1.set_ylabel("X1")
    ax1.set_zlabel("X2")
    ax1.legend()
    
    # --- Unit disk plot ---
    ax2 = fig.add_subplot(122)
    ax2.plot(points_disk[:,0], points_disk[:,1], 'bo', label='Mapped points')
    ax2.set_xlim([-1.1,1.1])
    ax2.set_ylim([-1.1,1.1])
    ax2.set_aspect('equal')
    ax2.set_title("Unit Disk Mapping")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    # draw disk boundary
    circle = plt.Circle((0,0),1.0,color='black',fill=False)
    ax2.add_patch(circle)
    ax2.legend()
    
    plt.show()


