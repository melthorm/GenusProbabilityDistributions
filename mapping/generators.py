
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd

#-----------------------------------------------
# Möbius transformation helpers
#-----------------------------------------------
def apply_mobius(z, M):
    a,b,c,d = M.flatten()
    return (a*z + b) / (c*z + d)

def mobius_from_3pts(src_pts, tgt_pts):
    """Compute Möbius map M (2x2) s.t. M(src[i])=tgt[i] for 3 complex pairs."""
    A = []
    for z,w in zip(src_pts, tgt_pts):
        A.append([z, 1.0, -w*z, -w])
    A = np.array(A, dtype=np.complex128)
    _,_,Vh = svd(A)
    x = Vh.conj().T[:,-1]
    M = x.reshape((2,2))
    return M

def project_to_SU11(M):
    """Force matrix to satisfy M*† J M = J (unit disk isometry)."""
    J = np.array([[1,0],[0,-1]], dtype=np.complex128)
    H = M.conj().T @ J @ M
    lam = H[0,0].real
    if lam <= 0:
        raise ValueError("λ ≤ 0: not circle-preserving; check input points.")
    M = M / np.sqrt(lam)
    detM = np.linalg.det(M)
    M = M / (detM**0.5)
    return M

#-----------------------------------------------
# Hyperbolic geodesic: circle orthogonal to unit disk
#-----------------------------------------------
def hyperbolic_circle_through(z1, z2):
    """Return center and radius of circle passing through z1,z2 orthogonal to unit circle."""
    x1, y1 = z1.real, z1.imag
    x2, y2 = z2.real, z2.imag
    # Solve center (cx,cy) from orthogonality and circle passing through z1,z2
    # Center formula for orthogonal circle
    denom = 2*(x1*y2 - y1*x2)
    if np.abs(denom) < 1e-12:  # nearly diameter
        return 0+0j, 1.0  # treat as straight line through center
    cx = ((x1**2 + y1**2 + 1)*(y2 - y1) - (x2**2 + y2**2 + 1)*(y1 - y2))/denom
    cy = ((x2**2 + y2**2 + 1)*(x1 - x2) - (x1**2 + y1**2 + 1)*(x2 - x1))/denom
    C = cx + 1j*cy
    R = abs(z1 - C)
    return C, R

def sample_hyperbolic_arc(z1, z2, N=50):
    C,R = hyperbolic_circle_through(z1, z2)
    if abs(C) < 1e-12:  # line through center
        return np.linspace(z1, z2, N)
    theta1 = np.angle(z1 - C)
    theta2 = np.angle(z2 - C)
    # ensure interpolation follows correct orientation
    if theta2 < theta1:
        theta2 += 2*np.pi
    angles = np.linspace(theta1, theta2, N)
    return C + R*np.exp(1j*angles)

#-----------------------------------------------
# Regular 4g-gon
#-----------------------------------------------
def regular_4g_vertices(g, phase=0.0):
    n = 4*g
    return np.exp(1j*(phase + 2*np.pi*np.arange(n)/n))

def side_endpoints(vertices, idx):
    n = len(vertices)
    return vertices[idx % n], vertices[(idx+1) % n]

#-----------------------------------------------
# Build one deck transformation generator
#-----------------------------------------------
g = 1
verts = regular_4g_vertices(g)
side_idx = 0

# source side
z1, z2 = side_endpoints(verts, side_idx)
z3 = 0.0 + 0.0j  # center

# target outside-triangle apex
mid_angle = np.angle(z1 + z2)
w3 = np.exp(1j*mid_angle)  # apex on unit circle outside polygon
tgt_pts = [z2, z1, w3]     # flip to match orientation
src_pts = [z1, z2, z3]
print(tgt_pts)
print(src_pts)

# Möbius generator
M = mobius_from_3pts(src_pts, tgt_pts)
M = project_to_SU11(M)

# Verify mapping
check_img = apply_mobius(np.array(src_pts), M)
print("Max mapping error:", np.max(np.abs(check_img - np.array(tgt_pts))))

# Five interior points
radii = [0.2, 0.35, 0.5, 0.65, 0.8]
angles = np.linspace(0.1, 1.2, 5)
points = np.array([r*np.exp(1j*a) for r,a in zip(radii, angles)])
mapped = apply_mobius(points, M)

def hyperbolic_arc(p, q, N=100):
    # Straight line if nearly collinear with origin
    if np.isclose(np.cross([p.real, p.imag], [q.real, q.imag]), 0):
        return np.linspace(p, q, N)
    
    # Compute center of circle orthogonal to unit circle passing through p and q
    # Formula from Poincare disk geometry
    a, b = p, q
    denom = a*np.conj(b) - np.conj(a)*b
    if np.isclose(denom, 0):
        return np.linspace(a, b, N)
    c = (a*abs(b)**2 - b*abs(a)**2)/denom
    r = abs(a - c)
    
    theta1, theta2 = np.angle(a-c), np.angle(b-c)
    # shortest arc
    if (theta2 - theta1) % (2*np.pi) > np.pi:
        theta1, theta2 = theta2, theta1 + 2*np.pi
    thetas = np.linspace(theta1, theta2, N)
    return c + r*np.exp(1j*thetas)

#-----------------------------------------------
# Plotting
#-----------------------------------------------
plt.figure(figsize=(7,7))
t = np.linspace(0, 2*np.pi, 400)
plt.plot(np.cos(t), np.sin(t), 'k-', lw=1)  # unit circle

# draw n-gon with curved hyperbolic arcs
for k in range(len(verts)):
    p, q = verts[k], verts[(k+1)%len(verts)]
    arc = hyperbolic_arc(p, q, N=100)
    print(arc)
    plt.plot(arc.real, arc.imag, 'gray', lw=1)

# source triple (red)
plt.scatter([z1.real, z2.real, z3.real],
            [z1.imag, z2.imag, z3.imag],
            color='red', s=70, label='source triple')

# target triple (blue)
plt.scatter([tgt_pts[0].real, tgt_pts[1].real, tgt_pts[2].real],
            [tgt_pts[0].imag, tgt_pts[1].imag, tgt_pts[2].imag],
            color='blue', s=70, label='target triple')

# interior points and images
plt.scatter(points.real, points.imag, color='black', marker='o', s=50, label='original 5 pts')
plt.scatter(mapped.real, mapped.imag, color='green', marker='x', s=70, label='mapped pts')

# arrows
for z, w in zip(points, mapped):
    plt.annotate('', xy=(w.real,w.imag), xytext=(z.real,z.imag),
                 arrowprops=dict(arrowstyle='->', lw=1.0, color='green'))

plt.gca().set_aspect('equal')
plt.xlim([-1.05,1.05])
plt.ylim([-1.05,1.05])
plt.title("Genus g=1: interior sector → outside triangle (curved hyperbolic sides)")
plt.legend(loc='lower left')
plt.show()

