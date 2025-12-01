import numpy as np
import matplotlib.pyplot as plt
from generators import HyperbolicPolygon, HyperbolicGenerators


def track_orbit_matrices(start, matrices, apply_fn):
    """Track orbit using an explicit list of matrices (or any callable maps)."""
    z = start
    out = [z]
    for M in matrices:
        z = apply_fn(z, M)
        out.append(z)
    return np.array(out)


def visualize_polygon_mapping(poly, mapped_poly, n_points=100):
    plt.figure(figsize=(6,6))
    ax = plt.gca()
    circle = plt.Circle((0,0), 1, color='k', fill=False, lw=1)
    ax.add_artist(circle)

    def plot_poly(verts, color, lw):
        N = len(verts)
        for i in range(N):
            z1, z2 = verts[i], verts[(i+1)%N]
            ax.plot([z1.real, z2.real], [z1.imag, z2.imag], color=color, lw=lw)

    plot_poly(poly.vertices, 'b', 3)       # original polygon in blue
    plot_poly(mapped_poly, 'r', 1)         # mapped polygon in red

    ax.set_aspect('equal')
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    plt.title("Original (blue) and Mapped (red) Polygon")
    plt.show()

    for i, (z_orig, z_mapped) in enumerate(zip(poly.vertices, mapped_poly)):
        print(f"Vertex {i}: start = {z_orig}, end = {z_mapped}, |start-end| = {abs(z_orig - z_mapped)}")


if __name__ == "__main__":
    p, q = 8, 8
    poly = HyperbolicPolygon(p, q)
    poly.compute_interior_points()
    gens = HyperbolicGenerators(poly)

    # sequences as indices
    seq_a1 = [6, 3, 0]
    seq_b1 = [5, 2, 7]
    seq_a2 = [2, 7, 4]
    seq_b2 = [1, 6, 3]

    # Build the commutator-style combined sequence with **true inverses**
    combined_matrices = []

    # a1 b1 a1^-1 b1^-1
    for idx in seq_a1:
        combined_matrices.append(gens.generators[idx])
    for idx in seq_b1:
        combined_matrices.append(gens.generators[idx])
    for idx in reversed(seq_a1):
        combined_matrices.append(np.linalg.inv(gens.generators[idx]))
    for idx in reversed(seq_b1):
        combined_matrices.append(np.linalg.inv(gens.generators[idx]))

    # a2 b2 a2^-1 b2^-1
    for idx in seq_a2:
        combined_matrices.append(gens.generators[idx])
    for idx in seq_b2:
        combined_matrices.append(gens.generators[idx])
    for idx in reversed(seq_a2):
        combined_matrices.append(np.linalg.inv(gens.generators[idx]))
    for idx in reversed(seq_b2):
        combined_matrices.append(np.linalg.inv(gens.generators[idx]))

    # Track a single vertex to see the orbit
    start_vertex = 0
    pts = track_orbit_matrices(poly.vertices[start_vertex], combined_matrices, gens.apply)
    print(f"Vertex {start_vertex}: start = {pts[0]}, end = {pts[-1]}, |start-end| = {abs(pts[0]-pts[-1])}")

    # Map entire polygon
    mapped_poly = poly.vertices
    for M in combined_matrices:
        mapped_poly = [gens.apply(z, M) for z in mapped_poly]

    # Visualize original vs mapped polygon
    visualize_polygon_mapping(poly, mapped_poly)

