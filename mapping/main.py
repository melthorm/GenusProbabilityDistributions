import numpy as np
import matplotlib.pyplot as plt
from generators import HyperbolicPolygon

def main():
    p, q = 8, 8
    poly = HyperbolicPolygon(p=p, q=q)
    poly.compute_interior_points()
    poly.compute_all_generators()

    # Starting point strictly inside fundamental polygon (e.g., polygon center)
    z = np.mean(poly.vertices)

    # Define a sequence of generators to apply (indices of generators)
    # Positive index = generator, negative index = inverse
    sequence = [2, 1, 1, 2, 2, 3]  

    points = [z]  # record trajectory
    for idx in sequence:
        if idx >= 0:
            G = poly.generators[idx]
        else:
            G = poly.generators_inv[-idx]
        z = poly.apply_generator(z, G)
        points.append(z)

    points = np.array(points)




    # Store layers separately
    layers = []
    layers.append([poly.vertices])  # layer 0: fundamental polygon

    n_layers = 2  # number of layers to generate
    for layer_index in range(1, n_layers+1):
        prev_layer = layers[-1]
        next_layer = []
        for G in poly.generators:
            for H in [G, np.linalg.inv(G)]:
                for vs in prev_layer:
                    next_layer.append([poly.apply_generator(z, H) for z in vs])
        layers.append(next_layer)

    # Visualize each layer individually
    for i, layer in enumerate(layers):
        print(f"Visualizing layer {i}, {len(layer)} polygons")
        poly.visualize(transformed_polygons=layer, n_points=150)
        plt.show();

    poly.visualize(transformed_polygons=layer, n_points=150)

    # Overlay trajectory
    plt.plot(points.real, points.imag, 'o-', color='orange', markersize=6, lw=2)
    plt.title("Trajectory of a point under generator sequence")
    plt.show()

if __name__ == "__main__":
    main()

