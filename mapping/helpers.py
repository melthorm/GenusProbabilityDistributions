import numpy as np
# --------------------
# Basic hyperbolic / Lorentz helpers
# --------------------

def minkowski_dot(a, b):
    return -a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def hyperbolic_geodesic_interp(V1, V2, t):
    """Geodesic interpolation between two hyperboloid points V1,V2."""

    dot = V1[0]*V2[0] - V1[1]*V2[1] - V1[2]*V2[2]
    d = np.arccosh(dot)
    if np.isnan(d) or d == 0:
        return V1.copy()
    X = (np.sinh((1-t)*d) / np.sinh(d)) * V1 + (np.sinh(t*d) / np.sinh(d)) * V2
    # Reproject to hyperboloid surface
    X[0] = np.sqrt(1.0 + X[1]**2 + X[2]**2)
    return X


def lorentz_cross(a, b):
    # Lorentzian cross product producing a vector orthogonal (in Minkowski sense)
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        -(a[1]*b[0] - a[0]*b[1])
    ])

def norm_spatial(v):
    # positive norm for tangent/spatial vectors: sqrt(minkowski_dot(v,v))
    val = minkowski_dot(v, v)
    if val <= 0:
        return 0.0
    return np.sqrt(val)

def dH(x, y):
    # hyperbolic distance with clamping to avoid numerical domain errors
    val = -minkowski_dot(x, y)
    if val < 1.0:
        val = 1.0
    return np.arccosh(val)

def exp_map(c, v):
    """Exponential map at c: v is tangent vector in R^3 (T_c H^2), returns point on hyperboloid."""
    norm_v = np.sqrt(np.dot(v, v))  # Euclidean norm of v in R^3 equals positive sqrt(minkowski_dot)
    if norm_v == 0.0:
        return c.copy()
    return np.cosh(norm_v)*c + np.sinh(norm_v)*(v / norm_v)

def log_map(c, x):
    """Logarithmic map: returns tangent vector w at c s.t. exp_map(c,w)=x."""
    # compute hyperbolic distance d
    d = dH(c, x)
    if d == 0.0:
        return np.zeros(3)
    # w = d * (x + <c,x> c) / sinh(d)
    return d * (x + minkowski_dot(c, x) * c) / np.sinh(d)

# --------------------
# Tangent basis at c
# --------------------

def tangent_basis(c):
    """
    Build an orthonormal (w.r.t. Minkowski->positive definite on tangent) basis e1,e2 of T_c H^2.
    We use two seed vectors in R^3 then Gram-Schmidt in Minkowski inner product.
    """
    # seeds that are generically linearly independent from c
    seed1 = np.array([0.0, 1.0, 0.0])
    seed2 = np.array([0.0, 0.0, 1.0])

    # project seeds to tangent (Minkowski-orthogonal to c)
    def proj_to_tangent(seed):
        # subtract component along c: seed - (<seed,c>/<c,c>)*c ; note <c,c> = -1
        return seed - (minkowski_dot(seed, c) / minkowski_dot(c, c)) * c

    u1 = proj_to_tangent(seed1)
    # normalize u1 using positive Minkowski norm
    n1 = np.sqrt(abs(minkowski_dot(u1, u1)))
    if n1 == 0:
        raise RuntimeError("failed to build tangent basis (degenerate seed1)")
    e1 = u1 / n1

    # second seed orthogonalize against c and e1 (Minkowski projections)
    u2 = proj_to_tangent(seed2)
    # subtract projection onto e1 (Minkowski)
    proj_e1_coeff = minkowski_dot(u2, e1) / minkowski_dot(e1, e1)
    u2 = u2 - proj_e1_coeff * e1

    n2 = np.sqrt(abs(minkowski_dot(u2, u2)))
    if n2 == 0:
        # rarely happens if seed2 collinear; pick a different seed
        seed2b = np.array([1.0, 0.0, 0.0])
        u2 = proj_to_tangent(seed2b)
        proj_e1_coeff = minkowski_dot(u2, e1) / minkowski_dot(e1, e1)
        u2 = u2 - proj_e1_coeff * e1
        n2 = np.sqrt(abs(minkowski_dot(u2, u2)))
        if n2 == 0:
            raise RuntimeError("failed to build tangent basis (degenerate seed2)")
    e2 = u2 / n2

    return e1, e2

# --------------------
# Polygon geometry on hyperboloid
# --------------------



def build_regular_polygon(c, e1, e2, two_n, radius):
    """
    Build a 'regular' 2n-gon by exponentiating radius * directions equally spaced in angle.
    The first vertex is aligned so that it lies along y = 0 in tangent plane (positive x-axis).
    """
    vertices = []
    for k in range(two_n):
        # start first vertex along e1 (x-axis), then equally space around circle
        ang = 2*np.pi * k / two_n  
        dir_t = np.cos(ang) * e1 + np.sin(ang) * e2
        v = exp_map(c, radius * dir_t)
        vertices.append(v)
    return vertices




def compute_side_normals(vertices):
    """
    For each edge [v_k, v_{k+1}], compute Lorentz-normal N_k to the plane containing that geodesic.
    Normalize so that minkowski_dot(N_k, N_k) = 1 (space-like normal).
    """
    N = []
    m = len(vertices)
    for k in range(m):
        a = vertices[k]
        b = vertices[(k+1) % m]
        nvec = lorentz_cross(a, b)  # orthogonal to both a and b in Minkowski sense
        # normalize: minkowski_dot(nvec,nvec) should be positive for spacelike normal
        nnorm_sq = minkowski_dot(nvec, nvec)
        if nnorm_sq <= 0:
            # numerical fallback — take absolute
            nnorm = np.sqrt(abs(nnorm_sq)) if nnorm_sq != 0 else 1.0
        else:
            nnorm = np.sqrt(nnorm_sq)
        nvec = nvec / nnorm
        N.append(nvec)
    return N

# --------------------
# Intersection distance s_max along unit tangent u: solves <N, cosh(s)c + sinh(s) u> = 0
# This yields tanh(s) = -<N,c> / <N,u>  -> s = arctanh(val) provided 0 < val < 1
# --------------------

def intersection_distance_for_normal(c, u, N):
    denom = minkowski_dot(N, u)
    if denom == 0:
        return None
    val = -minkowski_dot(N, c) / denom
    # require 0 < val < 1 to correspond to intersection in forward ray with positive s
    if val <= 0 or val >= 1:
        return None
    return np.arctanh(val)  # arctanh in numpy exists

def s_max_along_direction(c, u, normals):
    # returns smallest positive s_max among normals (first boundary hit)
    candidates = []
    for N in normals:
        s = intersection_distance_for_normal(c, u, N)
        if s is not None and s > 1e-15:
            candidates.append(s)
    if not candidates:
        # no intersection found (shouldn't happen for interior points)
        return None
    return min(candidates)

# --------------------
# Forward mapping f: H^2 -> D^2 (returns (x,y))
# --------------------

def forward_map(X, c, e1, e2, vertices, normals, two_n):
    # 1) compute log at c
    w = log_map(c, X)
    s = np.sqrt(np.dot(w, w))
    if s == 0:
        return np.array([0.0, 0.0])  # center maps to origin
    u = w / s  # unit tangent vector in R^3

    # Computes tangent vector from center to point X


    # 2) compute direction angle phi in tangent basis
    phi = np.arctan2(np.dot(w, e2), np.dot(w, e1))  # in (-pi, pi]

    # Computes the correct sector by projecting onto the unit basis

    # compute phi_k list for vertices
    phi_list = []
    for v in vertices:
        wv = log_map(c, v)
        phiv = np.arctan2(np.dot(wv, e2), np.dot(wv, e1))
        phi_list.append(phiv)
    # Get the actual sector

    # ensure phi_list sorted cyclically; they should be increasing around circle
    # find sector k with phi in [phi_k, phi_{k+1})
    # normalize angles to continuous range with phi0 as reference
    ref = phi_list[0]
    phi_norm = normalize_angle_to_interval(phi, ref)
    phi_list_norm = [normalize_angle_to_interval(p, ref) for p in phi_list]
    # sort with indices
    ordered = sorted(enumerate(phi_list_norm), key=lambda it: it[1])
    idxs, sorted_phis = zip(*ordered)
    # find insertion point
    k = find_sector_index(phi_norm, list(sorted_phis))
    idx_k = idxs[k]
    idx_k1 = idxs[(k+1) % two_n]
    phi_k = phi_list[idx_k]
    phi_k1 = phi_list[idx_k1]

    # fractional lambda
    phi_k_n = normalize_angle_to_interval(phi_k, ref)
    phi_k1_n = normalize_angle_to_interval(phi_k1, ref)
    if phi_k1_n <= phi_k_n:
        phi_k1_n += 2*np.pi
    phin = normalize_angle_to_interval(phi, ref)
    if phin < phi_k_n:
        phin += 2*np.pi
    # Computes lambda value that represents fractional value from the sector start thingy
    lam = (phin - phi_k_n) / (phi_k1_n - phi_k_n)
    lam = np.clip(lam, 0.0, 1.0)

    # compute s_max along unit tangent u
    smax = s_max_along_direction(c, u, normals)
    if smax is None:
        raise RuntimeError("No boundary intersection found for this direction (point maybe outside polygon)")

    # Normalize radial coordinate
    r = s / smax
    r = np.clip(r, 0.0, 1.0)

    # theta_k targets equally spaced around circle
    theta_k = (np.pi * idx_k) / (two_n // 2)  # recall two_n = 2n
    theta_k1 = (np.pi * idx_k1) / (two_n // 2)
    # fix wrap for theta interpolation
    tk = theta_k
    tk1 = theta_k1
    if tk1 <= tk:
        tk1 += 2*np.pi
    theta = (1-lam)*tk + lam*tk1

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.array([x, y]), np.array([r, theta])

# --------------------
# Inverse mapping: D^2 -> H^2
# --------------------

def inverse_map(z, c, e1, e2, vertices, normals, two_n):
    x, y = z
    rho = np.sqrt(x*x + y*y)
    if rho == 0:
        return c.copy()
    # tehta of hyperbola
    Theta = np.arctan2(y, x)

    # find sector k by Theta
    # target theta_k are equally spaced: theta_k = pi*k/n with two_n = 2n
    theta_list = [ (np.pi * k) / (two_n // 2) for k in range(two_n) ]

    # plcace all between 0 and 2 pi
    ref = theta_list[0]
    theta_list_norm = [normalize_angle_to_interval(t, ref) for t in theta_list]

    # sort
    ordered = sorted(enumerate(theta_list_norm), key=lambda it: it[1])
    idxs, sorted_thetas = zip(*ordered)
    # place between 0 and 2 pi
    Theta_n = normalize_angle_to_interval(Theta, ref)
    # find which interval normalized hyperbolic theta is corresponding to our intervals
    k = find_sector_index(Theta_n, list(sorted_thetas))
    idx_k = idxs[k]
    idx_k1 = idxs[(k+1) % two_n]

    # Our two sectors that hyperbolci theta lies between
    tk = theta_list[idx_k]
    tk1 = theta_list[idx_k1]

    # normalize for interpolation the two sectors (redundant)
    tk_n = normalize_angle_to_interval(tk, ref)
    tk1_n = normalize_angle_to_interval(tk1, ref)
    # should be 2 pi spaced apart (ie. scale them to be 2 pi spaced apart)
    if tk1_n <= tk_n:
        tk1_n += 2*np.pi
    Theta_adj = Theta_n
    if Theta_adj < tk_n:
        Theta_adj += 2*np.pi

    # Scale the theta of hyperbolic by the current sector 
    lam = (Theta_adj - tk_n) / (tk1_n - tk_n)
    lam = np.clip(lam, 0.0, 1.0)

    # interpolate direction phi from phi_k, phi_k1 (computed from vertices)
    phi_list = []
    for v in vertices:
        wv = log_map(c, v)
        phiv = np.arctan2(np.dot(wv, e2), np.dot(wv, e1))
        phi_list.append(phiv)
    phi_k = phi_list[idx_k]
    phi_k1 = phi_list[idx_k1]

    # normalize and interpolate
    refp = phi_list[0]
    pk = normalize_angle_to_interval(phi_k, refp)
    pk1 = normalize_angle_to_interval(phi_k1, refp)
    if pk1 <= pk:
        pk1 += 2*np.pi
    ph = (1-lam)*pk + lam*pk1

    # make unit tangent u from ph
    u = np.cos(ph)*e1 + np.sin(ph)*e2

    # compute s_max along this u
    smax = s_max_along_direction(c, u, normals)
    if smax is None:
        raise RuntimeError("No boundary intersection found for this direction in inverse")

    s = rho * smax
    # map back by exp_map
    p = exp_map(c, s * u)
    return p

# --------------------
# Angle helpers
# --------------------

def normalize_angle_to_interval(a, ref):
    # map a to [ref, ref+2pi)
    twopi = 2*np.pi
    out = a
    while out < ref:
        out += twopi
    while out >= ref + twopi:
        out -= twopi
    return out

def find_sector_index(angle, sorted_angles):
    """
    Given a scalar angle and a list of sorted angles (ascending),
    return index k such that angle in [sorted_angles[k], sorted_angles[k+1])
    with cyclic wrap.
    """
    n = len(sorted_angles)
    # ensure angle in range [first, first+2pi)
    first = sorted_angles[0]
    ang = angle
    if ang < first:
        ang += 2*np.pi
    for i in range(n):
        a0 = sorted_angles[i]
        a1 = sorted_angles[(i+1) % n]
        a1_adj = a1
        if a1_adj <= a0:
            a1_adj += 2*np.pi
        if a0 <= ang < a1_adj:
            return i
    return n-1



