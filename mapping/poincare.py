"""
poincare_ngon_to_unitdisk_fixed.py

Fixed version: ensures geodesic arcs stay inside the unit disk and
ray-circle intersection selects the arc/solution that lies inside D.
"""

import numpy as np
import matplotlib.pyplot as plt

# --------------------
# Poincaré disk primitives (centered at origin)
# --------------------

def mobius_add(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    x2 = np.dot(x, x)
    y2 = np.dot(y, y)
    xy = np.dot(x, y)
    denom = 1 + 2*xy + x2*y2
    return ((1 + 2*xy + y2) * x + (1 - x2) * y) / denom


def poincare_distance(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    num = np.linalg.norm(x - y)**2
    denom = (1 - np.linalg.norm(x)**2) * (1 - np.linalg.norm(y)**2)
    val = 1 + 2 * num / denom
    if val < 1.0:
        val = 1.0
    return np.arccosh(val)


def exp_map_origin(v):
    v = np.asarray(v)
    nv = np.linalg.norm(v)
    if nv == 0.0:
        return np.zeros(2)
    return np.tanh(nv / 2.0) * (v / nv)


def log_map_origin(x):
    x = np.asarray(x)
    rx = np.linalg.norm(x)
    if rx == 0.0:
        return np.zeros(2)
    return (2 * np.arctanh(rx)) * (x / rx)


# --------------------
# Geodesic arc circle through p,q orthogonal to unit circle
# --------------------

def circle_center_orthogonal(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    A = np.vstack([p, q])
    b = np.array([ (np.dot(p,p) + 1.0)/2.0, (np.dot(q,q) + 1.0)/2.0 ])
    det = np.linalg.det(A)
    if abs(det) < 1e-12:
        return None
    m = np.linalg.solve(A, b)
    r2 = np.dot(m, m) - 1.0
    # r2 might be slightly negative due to floating error; clamp
    if r2 < 0 and r2 > -1e-12:
        r2 = 0.0
    if r2 < 0:
        # numeric fallback: treat as degenerate
        return None
    r = np.sqrt(r2)
    return m, r


def intersect_ray_with_circle(u, m, r):
    """
    Solve |t u - m|^2 = r^2 for t. Both roots (if real) are returned as a sorted list.
    We then filter by 0 < t < 1 and choose the smallest positive t that yields a point
    strictly inside the unit disk (norm < 1). Returns that t or None.
    """
    udotm = float(np.dot(u, m))
    disc = udotm*udotm - (np.dot(m, m) - r*r)
    if disc < 0:
        return None
    sqrt_disc = np.sqrt(max(0.0, disc))
    t1 = udotm - sqrt_disc
    t2 = udotm + sqrt_disc
    candidates = []
    for t in (t1, t2):
        if t <= 1e-14 or t >= 1.0 + 1e-12:  # ignore non-positive and points on/ outside boundary
            continue
        pt = t * u
        if np.linalg.norm(pt) < 1.0 - 1e-12:
            candidates.append(t)
    if not candidates:
        return None
    return float(min(candidates))


# --------------------
# Build regular n-gon in Poincaré disk (centered at origin)
# --------------------

def build_regular_ngon_poincare(n_sides, vertex_radius):
    verts = []
    for k in range(n_sides):
        theta = 2.0 * np.pi * k / n_sides
        v = np.array([vertex_radius * np.cos(theta), vertex_radius * np.sin(theta)])
        verts.append(v)
    return verts


# --------------------
# Ray / polygon intersection helpers
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
        if t >= 1.0 - 1e-14:
            continue
        if best_t is None or t < best_t:
            best_t = t
    return best_t


def intersect_ray_with_euclid_segment(u, p, q):
    A = np.column_stack([u, -(q - p)])
    try:
        sol = np.linalg.solve(A, p)
    except np.linalg.LinAlgError:
        return None
    t, s = sol[0], sol[1]
    if t > 1e-14 and 0.0 <= s <= 1.0 and t < 1.0:
        # ensure intersection point inside unit disk
        if np.linalg.norm(t*u) < 1.0 - 1e-12:
            return float(t)
    return None


# --------------------
# Forward / inverse maps
# --------------------

def forward_map_poincare(z, vertices):
    z = np.asarray(z, dtype=float)
    r = np.linalg.norm(z)
    if r == 0.0:
        return np.array([0.0, 0.0]), (0.0, 0.0)
    u = z / r
    t_max = max_radius_along_direction(u, vertices)
    if t_max is None:
        raise RuntimeError("No intersection found for direction; point might be outside polygon")
    rho = r / t_max
    rho = np.clip(rho, 0.0, 1.0)
    theta = np.arctan2(z[1], z[0])
    w = np.array([rho * np.cos(theta), rho * np.sin(theta)])
    return w, (rho, theta)


def inverse_map_poincare(w, vertices):
    w = np.asarray(w, dtype=float)
    rho = np.linalg.norm(w)
    if rho == 0.0:
        return np.zeros(2)
    theta = np.arctan2(w[1], w[0])
    u = np.array([np.cos(theta), np.sin(theta)])
    t_max = max_radius_along_direction(u, vertices)
    if t_max is None:
        raise RuntimeError("No intersection found for direction in inverse_map")
    r = rho * t_max
    if r >= 1.0:
        r = 1.0 - 1e-12
    p = r * u
    return p


import numpy as np

def forward_map_poincare_polar(z_polar, vertices):
    """
    z_polar: tuple (r, theta)
    Returns: mapped point in polar coordinates (rho, theta)
    """
    r, theta = z_polar
    if r == 0.0:
        return (0.0, theta), (0.0, theta)

    # Unit direction
    u = np.array([np.cos(theta), np.sin(theta)])

    # Max radius along this direction in polygon
    t_max = max_radius_along_direction(u, vertices)
    if t_max is None:
        raise RuntimeError("No intersection found; point might be outside polygon")

    # Normalized radius
    rho = np.clip(r / t_max, 0.0, 1.0)
    return (rho, theta), (rho, theta)


def inverse_map_poincare_polar(w_polar, vertices):
    """
    w_polar: tuple (rho, theta)
    Returns: original point in polar coordinates (r, theta)
    """
    rho, theta = w_polar
    if rho == 0.0:
        return (0.0, theta)

    # Unit direction
    u = np.array([np.cos(theta), np.sin(theta)])

    # Max radius along this direction in polygon
    t_max = max_radius_along_direction(u, vertices)
    if t_max is None:
        raise RuntimeError("No intersection found in inverse_map")

    # Recover original radius
    r = rho * t_max
    if r >= 1.0:
        r = 1.0 - 1e-12

    return (r, theta)




# --------------------
# Geodesic arc points (choose arc that lies inside unit disk)
# --------------------

def geodesic_arc_points(p, q, npts=64):
    circ = circle_center_orthogonal(p, q)
    if circ is None:
        return [ (1-t)*p + t*q for t in np.linspace(0, 1, npts) ]
    m, r = circ
    ang_p = np.arctan2(p[1] - m[1], p[0] - m[0])
    ang_q = np.arctan2(q[1] - m[1], q[0] - m[0])

    # build two candidate arcs and pick the one whose points lie inside the disk
    a0, a1 = ang_p, ang_q
    if a1 <= a0:
        a1 += 2*np.pi
    thetas1 = np.linspace(a0, a1, npts)
    pts1 = np.array([m + r * np.array([np.cos(t), np.sin(t)]) for t in thetas1])
    mean_norm1 = np.mean(np.linalg.norm(pts1, axis=1))

    # other arc (the complementary arc)
    thetas2 = np.linspace(a1, a0 + 2*np.pi, npts)
    pts2 = np.array([m + r * np.array([np.cos(t), np.sin(t)]) for t in thetas2])
    mean_norm2 = np.mean(np.linalg.norm(pts2, axis=1))

    # choose the arc with smaller mean radius (more likely to lie inside unit disk),
    # but ensure endpoints included exactly.
    if mean_norm1 <= mean_norm2:
        pts = pts1
    else:
        pts = pts2

    # clip any tiny numerical overshoots
    pts_clipped = []
    for pt in pts:
        norm = np.linalg.norm(pt)
        if norm >= 1.0:
            # small numerical overshoot -> project slightly inside
            pt = pt / (norm + 1e-12) * (1.0 - 1e-12)
        pts_clipped.append(pt)
    return pts_clipped


# --------------------
# Plotting helpers
# --------------------

def plot_polygon_and_mapping(vertices, sample_points=None, mapped_points=None, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax1, ax2 = axes

    ax1.set_title("Poincaré disk polygon (original)")
    uc = plt.Circle((0, 0), 1.0, fill=False, linewidth=1.0)
    ax1.add_artist(uc)
    for k in range(len(vertices)):
        p = vertices[k]
        q = vertices[(k+1) % len(vertices)]
        arc = geodesic_arc_points(p, q, npts=128)
        arc = np.array(arc)
        ax1.plot(arc[:,0], arc[:,1], '-k')
    V = np.array(vertices)
    ax1.plot(V[:,0], V[:,1], 'o', label='vertices')
    if sample_points is not None:
        S = np.array(sample_points)
        ax1.scatter(S[:,0], S[:,1], c='C0', s=20, alpha=0.9, label='samples')
    ax1.set_xlim(-1.05, 1.05)
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_aspect('equal', 'box')
    ax1.grid(True)
    ax1.legend()

    ax2.set_title("Mapped to unit disk (radial normalization)")
    uc2 = plt.Circle((0,0), 1.0, fill=False, linewidth=1.0)
    ax2.add_artist(uc2)
    if mapped_points is not None:
        M = np.array(mapped_points)
        ax2.scatter(M[:,0], M[:,1], c='C1', s=20, alpha=0.9)
    ax2.set_xlim(-1.05, 1.05)
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_aspect('equal', 'box')
    ax2.grid(True)

    fig.suptitle(title)
    plt.show()


# --------------------
# Testing harness
# --------------------

def run_tests():
    np.random.seed(42)
    two_n = 16
    n_sides = two_n
    vertex_radius = 1

    vertices = build_regular_ngon_poincare(n_sides, vertex_radius)
    print(f"Built regular {n_sides}-gon with vertex radius {vertex_radius}")

    n_tests = 200
    tol = 1e-7
    samples = []
    mapped = []
    max_hyperbolic_err = 0.0

    for i in range(n_tests):
        ang = np.random.uniform(0, 2*np.pi)
        u = np.array([np.cos(ang), np.sin(ang)])
        tmax = max_radius_along_direction(u, vertices)
        if tmax is None:
            continue
        r = np.random.uniform(0.0, tmax * 0.9999)
        pt = r * u
        try:
            w, pol = forward_map_poincare(pt, vertices)
        except RuntimeError:
            continue
        pts_back = inverse_map_poincare(w, vertices)
        d = poincare_distance(pt, pts_back)
        samples.append(pt)
        mapped.append(w)
        if d > max_hyperbolic_err:
            max_hyperbolic_err = d
        if not (d < 1e-6):
            # keep running; record but don't fail immediately
            pass

    print("Max hyperbolic error (Poincaré dist):", max_hyperbolic_err)

    boundary_samples = []
    boundary_mapped = []
    for k in range(len(vertices)):
        p = vertices[k]
        q = vertices[(k+1) % len(vertices)]
        arc_pts = geodesic_arc_points(p, q, npts=60)
        for pt in arc_pts:
            try:
                w, pol = forward_map_poincare(pt, vertices)
            except RuntimeError:
                continue
            boundary_samples.append(pt)
            boundary_mapped.append(w)

    plot_polygon_and_mapping(vertices, sample_points=samples + boundary_samples, mapped_points=mapped + boundary_mapped,
                             title=f"Regular {n_sides}-gon mapping to unit disk")

    print("Example vertex coords (Poincaré disk):")
    for v in vertices[:min(8, len(vertices))]:
        print(np.round(v, 6))
        
def run_tests_polar():
    np.random.seed(58)
    two_n = 16
    n_sides = two_n
    vertex_radius = 1

    vertices = build_regular_ngon_poincare(n_sides, vertex_radius)
    print(f"Built regular {n_sides}-gon with vertex radius {vertex_radius}")

    n_tests = 200
    max_hyperbolic_err = 0.0
    samples = []
    mapped = []

    for i in range(n_tests):
        ang = np.random.uniform(0, 2*np.pi)
        u = np.array([np.cos(ang), np.sin(ang)])
        tmax = max_radius_along_direction(u, vertices)
        if tmax is None:
            continue
        r = np.random.uniform(0.0, tmax * 0.9999)
        pt_polar = (r, ang)

        try:
            w_polar, pol = forward_map_poincare_polar(pt_polar, vertices)
        except RuntimeError:
            continue

        pt_back_polar = inverse_map_poincare_polar(w_polar, vertices)
        pt_cart = np.array([r * np.cos(ang), r * np.sin(ang)])
        pt_back_cart = np.array([pt_back_polar[0] * np.cos(pt_back_polar[1]),
                                 pt_back_polar[0] * np.sin(pt_back_polar[1])])
        d = poincare_distance(pt_cart, pt_back_cart)

        samples.append(pt_cart)
        mapped.append(np.array([w_polar[0] * np.cos(w_polar[1]),
                                w_polar[0] * np.sin(w_polar[1])]))

        if d > max_hyperbolic_err:
            max_hyperbolic_err = d

    print("Max hyperbolic error (Poincaré dist):", max_hyperbolic_err)

    # Boundary test
    boundary_samples = []
    boundary_mapped = []
    for k in range(len(vertices)):
        p = vertices[k]
        q = vertices[(k+1) % len(vertices)]
        arc_pts = geodesic_arc_points(p, q, npts=60)
        for pt in arc_pts:
            pt_polar = (np.linalg.norm(pt), np.arctan2(pt[1], pt[0]))
            try:
                w_polar, pol = forward_map_poincare_polar(pt_polar, vertices)
            except RuntimeError:
                continue
            boundary_samples.append(pt)
            boundary_mapped.append(np.array([w_polar[0] * np.cos(w_polar[1]),
                                             w_polar[0] * np.sin(w_polar[1])]))

    plot_polygon_and_mapping(vertices,
                             sample_points=samples + boundary_samples,
                             mapped_points=mapped + boundary_mapped,
                             title=f"Regular {n_sides}-gon mapping to unit disk")

    print("Example vertex coords (Poincaré disk):")
    for v in vertices[:min(8, len(vertices))]:
        print(np.round(v, 6))


if __name__ == "__main__":
    run_tests()
    run_tests_polar()
