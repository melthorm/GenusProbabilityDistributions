import numpy as np
import matplotlib.pyplot as plt

def poincare_arc_boundary(z1, z2, num_points=100):
    x1, y1 = z1
    x2, y2 = z2
    # Check if points are on a diameter
    if np.isclose(x1*y2 - x2*y1, 0):
        t = np.linspace(0, 1, num_points)
        return np.outer(1-t, z1) + np.outer(t, z2)
    
    # Circle center formula for arc orthogonal to unit circle
    d = 2*(x1*y2 - x2*y1)
    cx = ((x1**2 + y1**2 - 1)*(y2 - y1) - (x2**2 + y2**2 - 1)*(y2 - y1)) / d
    cy = ((x2**2 + y2**2 - 1)*(x2 - x1) - (x1**2 + y1**2 - 1)*(x2 - x1)) / d
    C = np.array([cx, cy])
    r = np.linalg.norm(C - z1)
    
    theta1 = np.arctan2(y1 - C[1], x1 - C[0])
    theta2 = np.arctan2(y2 - C[1], x2 - C[0])
    # Ensure the arc curves inward
    if theta2 < theta1:
        theta2 += 2*np.pi
    theta = np.linspace(theta1, theta2, num_points)
    
    x_arc = C[0] + r*np.cos(theta)
    y_arc = C[1] + r*np.sin(theta)
    return np.column_stack([x_arc, y_arc])

def poincare_ngon_edges(points):
    fig, ax = plt.subplots()
    ax.add_artist(plt.Circle((0,0),1,color='k',fill=False,linewidth=1.5))
    
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            arc = poincare_arc_boundary(points[i], points[j])
            ax.plot(arc[:,0], arc[:,1], 'r-')
    
    points = np.array(points)
    ax.plot(points[:,0], points[:,1], 'bo', markersize=8)
    ax.set_aspect('equal')
    ax.set_xlim([-1.05,1.05])
    ax.set_ylim([-1.05,1.05])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

# Example: 8 points on the circle like in your image
angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
points = [[np.cos(a), np.sin(a)] for a in angles]

poincare_ngon_edges(points)
