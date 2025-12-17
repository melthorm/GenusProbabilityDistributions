import numpy as np
from generators import HyperbolicPolygon, HyperbolicGenerators
from itertools import product

def enumerate_polygons_full(polygon: HyperbolicPolygon, generators: HyperbolicGenerators, depth=5):
    """
    This guy  just generates all possible sequences
    """
    num_generators = len(generators.generators)
    polygons_list = []

    # Enumerate all words of length 1..depth
    for d in range(1, depth+1):
        for seq in product(range(num_generators), repeat=d):
            transformed_vertices = polygon.vertices.copy()
            for idx in seq:
                M = generators.generators[idx]
                transformed_vertices = [generators.apply_matrix(z, M) for z in transformed_vertices]
            polygons_list.append((list(seq), transformed_vertices))
    print(f"Total sequences generated is {len(polygons_list)} polygons")
    return polygons_list

def unique_polygons(polygons_list, generators, tol=1e-14):
    
    seen_points = set()
    unique_list = []
    origin = 0 + 0j
    for seq, _ in polygons_list:
        # Compute image of the origin under this sequence
        z = origin
        for idx in seq:
            M = generators.generators[idx]
            z = generators.apply_matrix(z, M)

        # Use rounded coordinates for floating-point tolerance
        key = (round(z.real / tol) * tol, round(z.imag / tol) * tol)
        if key not in seen_points:
            seen_points.add(key)
            # Recompute transformed vertices if needed
            transformed_vertices = poly.vertices.copy()
            for idx in seq:
                M = generators.generators[idx]
                transformed_vertices = [generators.apply_matrix(v, M) for v in transformed_vertices]
            unique_list.append((seq, transformed_vertices))

    return unique_list

# Usage example
p, q = 8, 8
poly = HyperbolicPolygon(p, q)
gens = HyperbolicGenerators(poly)
poly.compute_interior_points()

depth = 5
polygons_list = enumerate_polygons_full(poly, gens, depth=depth)
unique_polygons_list = unique_polygons(polygons_list, gens)

print(f"Enumerated {len(unique_polygons_list)} polygons up to depth {depth}.")
# Print all unique sequences
#for seq, _ in unique_polygons_list:
    #print(seq)
