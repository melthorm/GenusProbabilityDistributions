"""
poincare_ngon_direct_r.py

Build a regular 4n-gon in the Poincaré disk using the closed-form formula for the
Euclidean circum-radius R (inside the unit disk) derived from the interior angle α.
Provide bijective forward/inverse radial normalization maps Poincaré-polygon <-> unit disk.

Usage:
    python poincare_ngon_direct_r.py
"""

import numpy as np
import matplotlib.pyplot as plt

# --------------------
# Closed-form Euclidean radius from interior angle alpha
# --------------------
def vertex_radius_from_alpha(n_sides, alpha):
    """
    Compute Euclidean circumradius R in the Poincaré disk for a regular n-sided polygon
    with interior angle `alpha` at each vertex.

    Formula (derived from hyperbolic triangle relations):
        R = sqrt( (cos(pi/n) - cos(alpha/2)) / (cos(pi/n) + cos(alpha/2)) )

    Notes:
      - alpha is the interior angle at each vertex (in radians).
      - The expression requires cos(pi/n) > cos(alpha/2) (i.e. the numerator positive).
      - For hyperbolic polygons alpha < pi*(1 - 2/n) is not required here; check domain.
    """
    # numeric guards
    if n_sides <= 2:
        raise ValueError("n_sides must be >= 3")
    c1 = np.cos(np.pi / n_sides)
    c2 = np.cos(alpha / 2.0)
    num = c1 - c2
    den = c1 + c2
    if den <= 0:
        raise ValueError("Denominator nonpositive in radius formula (invalid alpha/n).")
    if num <= 0:
        # polygon would push to or beyond boundary: return radius extremely close to 1
        return 1.0 - 1e-12
    R2 = num / den
    # numerical safety clamp
    if R2 < 0:
        R2 = 0.0
    R = np.sqrt(R2)
    # final clamp to be strictly < 1
    if R >= 1.0:
        R = 1.0 - 1e-12
    return float(R)


# --------------------
# Basic Poincaré helpers reused (minimal)
# --------------------

def circle_center_orthogonal(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    A = np.vstack([p, q])
    b = np.array([ (np.dot(p,p) + 1.0)/2.0, (np.dot(q,q) + 1.0)/2.0 ])
    det = np.linalg.det(A)
    if abs(det) < 1e-14:
        return None
    m = np.linalg.solve(A, b)
    r2 = np.dot(m, m) - 1.0
    # clamp slight negative noise
    if r2 < 0 and r2 > -1e-12:
        r2 = 0.0
    if r2 < 0:
        return None
    r = np.sqrt(r2)
    return m, r


def intersect_ray_with_circle(u, m, r):
    udotm = float(np.dot(u, m))
    disc = udotm*udotm - (np.dot(m, m) - r*r)
    if disc < 0:
        return None
    sqrt_disc = np.sqrt(max(0.0, disc))
    t1 = udotm - sqrt_disc
    t2 = udotm + sqrt_disc
    candidates = []
    for t in (t1, t2):
        if t <= 1e-14 or t >= 1.0 - 1e-12:
            continue
        pt = t * u
        if np.linalg.norm(pt) < 1.0 - 1e-12:
            candidates.append(t)
    if not candidates:
        return None
    return float(min(candidates))


def intersect_ray_with_euclid_segment(u, p, q):
    A = np.column_stack([u, -(q - p)])
    try:
        sol = np.linalg.solve(A, p)
    except np.linalg.LinAlgError:
        return None
    t, s = sol[0], sol[1]
    if t > 1e-14 and 0.0 <= s <= 1.0 and t < 1.0 and np.linalg.norm(t*u) < 1.0 - 1e-12:
        return float(t)
    return None


# --------------------
# Build regular n-gon using direct R formula
# --------------------

def build_regular_ngon_from_alpha(n_sides, alpha):
    """
    Build regular n-gon vertices in Poincaré disk using closed-form R computed from alpha.
    Returns (vertices, R).
    """
    R = vertex_radius_from_alpha(n_sides, alpha)
    verts = []
    for k in range(n_sides):
        theta = 2.0 * np.pi * k / n_sides
        v = np.array([R * np.cos(theta), R * np.sin(theta)])
        verts.append(v)
    return verts, R


# --------------------
# For a direction u compute t_max (same logic as earlier)
# --------------------

def max_radius_along_direction(u, vertices):
    best_t = None
    m = len(vertices)
    for k in range(m):
        p = vertices[k]
        q = vertices[(k+1) % m]
        circ = circle_center_orthogonal(p, q)
        if circ is None:
            t_line = intersect_ray_with_euclid_segment(u, p, q)
            if t_line is not None:
                if best_t is None or t_line < best_t:
                    best_t = t_line
            continue
        center, rad = circ
        t = intersect_ray_with_circle(u, center, rad)
        if t is None:
            continue
        if best_t is None or t < best_t:
            best_t = t
    return best_t


# --------------------
# Forward / inverse bijection (Poincaré polygon interior) <-> unit disk
# --------------------

def forward_bijection_poincare_to_unit(z, vertices):
    z = np.asarray(z, dtype=float)
    r = np.linalg.norm(z)
    if r == 0:
        return np.array([0.0, 0.0]), (0.0, 0.0)
    u = z / r
    t_max = max_radius_along_direction(u, vertices)
    if t_max is None:
        raise RuntimeError("direction does not intersect polygon boundary")
    rho = r / t_max
    rho = np.clip(rho, 0.0, 1.0)
    theta = np.arctan2(z[1], z[0])
    w = np.array([rho * np.cos(theta), rho * np.sin(theta)])
    return w, (rho, theta)


def inverse_bijection_unit_to_poincare(w, vertices):
    w = np.asarray(w, dtype=float)
    rho = np.linalg.norm(w)
    if rho == 0:
        return np.zeros(2)
    theta = np.arctan2(w[1], w[0])
    u = np.array([np.cos(theta), np.sin(theta)])
    t_max = max_radius_along_direction(u, vertices)
    if t_max is None:
        raise RuntimeError("direction does not intersect polygon boundary")
    r = rho * t_max
    if r >= 1.0:
        r = 1.0 - 1e-12
    p = r * u
    return p


# --------------------
# Utility: geodesic arc points for plotting
# --------------------

def geodesic_arc_points(p, q, npts=64):
    circ = circle_center_orthogonal(p, q)
    if circ is None:
        return [ (1-t)*p + t*q for t in np.linspace(0,1,npts) ]
    m, r = circ
    ang_p = np.arctan2(p[1]-m[1], p[0]-m[0])
    ang_q = np.arctan2(q[1]-m[1], q[0]-m[0])
    if ang_q <= ang_p:
        ang_q += 2*np.pi
    thetas = np.linspace(ang_p, ang_q, npts)
    pts = [ m + r * np.array([np.cos(t), np.sin(t)]) for t in thetas ]
    # clip minor numerical overshoot
    pts = [ (pt / max(1.0, np.linalg.norm(pt))) * (1.0 - 1e-12) if np.linalg.norm(pt) >= 1.0 else pt for pt in pts ]
    return pts


# --------------------
# Plot / test harness
# --------------------

def plot_polygon(vertices, samples=None, mapped=None, title=""):
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    uc = plt.Circle((0,0), 1.0, fill=False, linewidth=1.0)
    ax.add_artist(uc)
    for k in range(len(vertices)):
        p = vertices[k]
        q = vertices[(k+1) % len(vertices)]
        arc = geodesic_arc_points(p, q, npts=120)
        arc = np.array(arc)
        ax.plot(arc[:,0], arc[:,1], '-k', linewidth=1.0)
    V = np.array(vertices)
    ax.plot(V[:,0], V[:,1], 'o', label='vertices')
    if samples is not None:
        S = np.array(samples)
        ax.scatter(S[:,0], S[:,1], c='C0', s=16)
    if mapped is not None:
        M = np.array(mapped)
        # plot mapped points as well in another subplot for comparison
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect('equal', 'box')
    ax.grid(True)
    ax.set_title(title)
    plt.show()


def run_demo():
    # Choose polygon parameters
    two_n = 8                # 2n = number of sides; example 8-gon
    n_sides = two_n
    # interior vertex angle alpha in radians (choose something < pi)
    alpha = 1.0              # example interior angle (radians)
    vertices, R = build_regular_ngon_from_alpha(n_sides, alpha)
    print(f"Built regular {n_sides}-gon with interior angle alpha={alpha:.6f} rad -> Euclidean R={R:.6f}")

    # sample interior points and test bijection
    samples = []
    mapped = []
    recovered = []
    n_tests = 200
    max_err = 0.0
    for _ in range(n_tests):
        ang = np.random.uniform(0, 2*np.pi)
        u = np.array([np.cos(ang), np.sin(ang)])
        tmax = max_radius_along_direction(u, vertices)
        if tmax is None:
            continue
        r = np.random.uniform(0.0, tmax * 0.9999)
        p = r * u
        w, pol = forward_bijection_poincare_to_unit(p, vertices)
        p_back = inverse_bijection_unit_to_poincare(w, vertices)
        # measure Euclidean difference (small) and Poincaré distance (optionally)
        err = np.linalg.norm(p - p_back)
        if err > max_err:
            max_err = err
        samples.append(p)
        mapped.append(w)
        recovered.append(p_back)
    print("max euclidean reconstruction error:", max_err)

    # plot original polygon and sample points
    plot_polygon(vertices, samples=samples, title=f"Regular {n_sides}-gon in Poincaré disk (R={R:.4f})")

    # Also plot mapped points in unit disk for visual confirmation
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    uc = plt.Circle((0,0), 1.0, fill=False, linewidth=1.0)
    ax.add_artist(uc)
    if mapped:
        M = np.array(mapped)
        ax.scatter(M[:,0], M[:,1], c='C1', s=16)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect('equal', 'box')
    ax.set_title("Mapped points in unit disk")
    ax.grid(True)
    plt.show()


if __name__ == "__main__":
    run_demo()

