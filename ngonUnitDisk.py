# ngon_hyperboloid_to_disk.py
import numpy as np
from plotting import *
from helpers import *

# --------------------
# Test harness
# --------------------

def run_tests():
    #np.random.seed(1234)
    two_n = 6  # 2n polygon 
    c = np.array([1.0, 0.0, 0.0])  # center on hyperboloid
    e1, e2 = tangent_basis(c)

    # build a regular polygon with vertices at geodesic radius Rv
    Rv = 2  # geodesic distance from center to vertices (in tangent units)
    vertices = build_regular_polygon(c, e1, e2, two_n, Rv)
    normals = compute_side_normals(vertices)

    print(f"Testing mapping for a regular {two_n}-gon (Rv={Rv})\n")
    n_tests = 50
    tol = 1e-7
    # After generating random points and mapping
    points_hyperboloid = []
    points_disk = []
    for i in range(n_tests):
        # pick random tangent direction and radius within polygon (0 <= s < s_max for that direction)
        alpha = np.random.uniform(0, 2*np.pi)
        beta = np.random.uniform(0.0, 1.0)
        # for random direction form unit u
        u = np.cos(alpha) * e1 + np.sin(alpha) * e2
        smax = s_max_along_direction(c, u, normals)
        if smax is None:
            # skip directions that do not hit boundary (shouldn't happen for interior sector)
            continue
        s = np.random.uniform(0.0, smax * 0.999)  # ensure strictly inside
        X = exp_map(c, s * u)  # random interior point

        z, polars = forward_map(X, c, e1, e2, vertices, normals, two_n)  # disk coordinates
        X_back = inverse_map(z, c, e1, e2, vertices, normals, two_n)

        dist = dH(X, X_back)
        points_hyperboloid.append(X)
        points_disk.append(z)
        print(f"Test {i+1:2d}:")
        print(f"  Hyperboloid X      = {np.round(X, 6)}")
        print(f"  -> Disk z          = {np.round(z, 6)}")
        print(f"  -> In Polar        = {np.round(polars, 6)}")
        print(f"  <- X_back          = {np.round(X_back, 6)}")
        print(f"  hyperbolic error   = {dist:.3e}")
        if dist > tol:
            print("  >>> FAILED: distance exceeds tolerance")
            break
    else:
        print("\nAll tests passed within tolerance.")

    points_hyperbolid_boundary, points_disk_boundary = test_boundary_geodesic_mapping(vertices, c, e1, e2, normals, two_n) 
    #points_hyperboloid += points_hyperbolid_boundary
    #points_disk += points_disk_boundary
    plot_hyperboloid_and_disk(points_hyperboloid, points_disk, vertices=vertices, c=c)
    print(vertices)


def test_boundary_geodesic_mapping(vertices, c, e1, e2, normals, two_n):
    points_hyperboloid = []
    points_disk = []

    for i in range(len(vertices)):
        V1, V2 = vertices[i], vertices[(i+1) % len(vertices)]
        for t in np.linspace(0, 1, 20):
            X = hyperbolic_geodesic_interp(V1, V2, t)
            z, polars = forward_map(X, c, e1, e2, vertices, normals, two_n)
            X_back = inverse_map(z, c, e1, e2, vertices, normals, two_n)
            #print(f"Boundary polars: {np.round(polars, 6)}  |z|={np.linalg.norm(z):.6f}")
            #print(f"Hyperbolic points: {np.round(X, 6)}, {np.round(X_back, 6)}")
            points_hyperboloid.append(X)
            points_disk.append(z)

    plot_hyperboloid_and_disk(points_hyperboloid, points_disk, vertices=vertices, c=c)
    return points_hyperboloid, points_disk




if __name__ == "__main__":
    run_tests()

