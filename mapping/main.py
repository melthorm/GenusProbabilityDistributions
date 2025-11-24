def main():
    import numpy as np
    import matplotlib.pyplot as plt
    from generators import HyperbolicPolygon



    n = 4  # vertices
    r = 0.5  # radius inside unit disk
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    vertices = (r * np.exp(1j * angles)).tolist()

    side_pairs = [
        ((vertices[i], vertices[(i+1)%n]), (vertices[(i+4)%n], vertices[(i+5)%n]))
        for i in range(n//2)
    ]

    # Create polygon
    poly = HyperbolicPolygon(vertices, side_pairs)

    # Compute interior points for side-pairs
    poly.compute_interior_points()

    # Compute all generators
    poly.compute_all_generators()

    # Apply all generators and inverses to polygon vertices for visualization
    transformed_polys = []
    for G in poly.generators:
        for H in [G, np.linalg.inv(G)]:
            transformed_polys.append(poly.apply_generator_to_polygon(H))



    # Visualize polygon and all transformations
    poly.visualize(transformed_polygons=transformed_polys, n_points=150)

    transformed2_polys = []
    for G in poly.generators:
        for H in [G, np.linalg.inv(G)]:
            for vs in transformed_polys:
                transformed2_polys.append([poly.apply_generator(z, H) for z in vs])


    poly.visualize(transformed_polygons=transformed2_polys, n_points=150)


    transformed3_polys = []
    for G in poly.generators:
        for H in [G, np.linalg.inv(G)]:
            for vs in transformed2_polys:
                transformed3_polys.append([poly.apply_generator(z, H) for z in vs])


    poly.visualize(transformed_polygons=transformed3_polys, n_points=150)
    
    # Sample points strictly inside the polygon using random barycentric interpolation
    n_samples = 50
    sampled_points = []
    verts = np.array(vertices)
    N = len(verts)
    for _ in range(n_samples):
        # Pick two consecutive vertices to form a triangle with polygon center
        center = np.mean(verts)
        i = np.random.randint(N)
        v0 = verts[i]
        v1 = verts[(i+1)%N]
        # random barycentric coordinates inside the triangle
        s, t = np.random.uniform(0,1,2)
        if s + t > 1:
            s, t = 1-s, 1-t
        z = v0 + s*(v1 - v0) + t*(center - v0)
        sampled_points.append(z)
    sampled_points = np.array(sampled_points)

    # Apply the first generator to sampled points
    G0 = poly.generators[0]
    transformed_points = np.array([poly.apply_generator(z, G0) for z in sampled_points])
    plt.scatter(sampled_points.real, sampled_points.imag, color='orange', s=30, alpha=0.8, label='Sampled points')
    plt.scatter(transformed_points.real, transformed_points.imag, color='green', s=30, alpha=0.8, label='Transformed points')
    plt.legend()
    plt.title("Polygon, generator images, sampled points inside, and their images")
    plt.show()


if __name__ == "__main__":
    main()

