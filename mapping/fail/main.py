import numpy as np
import matplotlib.pyplot as plt
from generators import HyperbolicPolygon, HyperbolicGenerators
import sys

def track_orbit(poly, gens, start, seq):
    Glist = gens.generators
    z = start
    out = [z]
    for idx in seq:
        M = Glist[idx % len(Glist)]
        z = gens.apply(z, M)
        out.append(z)
    return np.array(out)


def build_layers(poly, gens, n_layers):
    Glist = gens.generators
    layers = [[poly.vertices]]
    for _ in range(n_layers):
        prev = layers[-1]
        nxt = []
        for M in Glist:
            Minv = np.linalg.inv(M)
            for vs in prev:
                nxt.append([gens.apply(z, M) for z in vs])
                nxt.append([gens.apply(z, Minv) for z in vs])
        layers.append(nxt)
    return layers


def inverse_sequence(seq, p):
    half = p // 2
    return [(x + half) % p for x in reversed(seq)]


def visualize_sequence(
    p,
    q,
    start_index,
    seq,
    name,
    n_layers=1,
    marker_color_start='red',
    marker_color_end='blue',
    marker_shape_start='o',
    marker_shape_end='o'
):
    poly = HyperbolicPolygon(p, q)
    poly.compute_interior_points()
    gens = HyperbolicGenerators(poly)

    pts = track_orbit(poly, gens, poly.vertices[start_index], seq)
    layers = build_layers(poly, gens, n_layers)

    poly.visualize(layers[-1], n_points=150)
    plt.plot(pts.real, pts.imag, '-', lw=2, color='black')

    plt.plot(
        pts[0].real,
        pts[0].imag,
        marker_shape_start,
        markersize=10,
        color=marker_color_start
    )
    plt.plot(
        pts[-1].real,
        pts[-1].imag,
        marker_shape_end,
        markersize=10,
        color=marker_color_end
    )

    plt.title(
        f"{name} at vertex {start_index} and starts at "
        f"{marker_color_start}, ends at {marker_color_end}"
    )
    plt.show()


def compose_sequences(*seqs):
    combined = []
    for seq in seqs:
        combined.extend(seq)
    return combined
    
def visPathPolygon(seq):
    p, q = 8, 8
    poly = HyperbolicPolygon(p, q)
    gens = HyperbolicGenerators(poly)

    journey = [poly.vertices]

    verts = poly.vertices
    for idx in seq:
        M = gens.generators[idx]
        verts = [gens.apply_matrix(z, M) for z in verts]
        journey.append(verts)

    plt.figure(figsize=(7,7))
    ax = plt.gca()
    circle = plt.Circle((0,0), 1, color='k', fill=False, lw=1)
    ax.add_artist(circle)

    colors = ['b','r','g','m','c','y','k']

    def plot_poly(verts, color):
        N = len(verts)
        for i in range(N):
            z1 = verts[i]
            z2 = verts[(i+1)%N]
            c = poly.compute_circle_center(z1, z2)
            if c is None:
                ax.plot([z1.real, z2.real], [z1.imag, z2.imag], color=color, lw=2)
            else:
                r = abs(z1 - c)
                t1 = np.angle(z1 - c)
                t2 = np.angle(z2 - c)
                d = t2 - t1
                if d < -np.pi: d += 2*np.pi
                elif d > np.pi: d -= 2*np.pi
                th = t1 + np.linspace(0,1,300)*d
                x = c.real + r*np.cos(th)
                y = c.imag + r*np.sin(th)
                ax.plot(x, y, color=color, lw=2)

    for i, verts in enumerate(journey):
        plot_poly(verts, colors[i % len(colors)])

    ax.set_aspect('equal')
    ax.set_xlim(-1.05,1.05)
    ax.set_ylim(-1.05,1.05)
    plt.show()


if __name__ == "__main__":
    p, q = 8, 8
    n_layers = 1

    seq_a1 = [2, 7, 4]
    #visualize_sequence(p, q, start_index=1, seq=seq_a1, n_layers=n_layers, name="a1")

    seq_b1 = [1, 6, 3]
    #visualize_sequence(p, q, start_index=0, seq=seq_b1, n_layers=n_layers, name="b1")

    a1_inv = [6, 1, 4]
    #visualize_sequence(p, q, start_index=6, seq=a1_inv, n_layers=n_layers, name="a1_inv")

    b1_inv = [5, 0, 3]
    #visualize_sequence(p, q, start_index=5, seq=b1_inv, n_layers=n_layers, name="b1_inv")

    seq_a2 = [6, 3, 0]
    #visualize_sequence(p, q, start_index=5, seq=seq_a2, n_layers=n_layers, name="a2")

    seq_b2 = [5, 2, 7]
    #visualize_sequence(p, q, start_index=4, seq=seq_b2, n_layers=n_layers, name="b2")

    a2_inv = [2, 5, 0]
    #visualize_sequence(p, q, start_index=2, seq=a2_inv, n_layers=n_layers, name="a2_inv")

    b2_inv = [1, 4, 7]
    #visualize_sequence(p, q, start_index=1, seq=b2_inv, n_layers=n_layers, name="b2_inv")

    
    combined = compose_sequences(
        seq_a1, seq_b1, inverse_sequence(a1_inv, p), inverse_sequence(b1_inv, p),
        seq_a2, seq_b2, inverse_sequence(a2_inv, p), inverse_sequence(b2_inv, p)
    )
    # 0 is t1, 1 is t2, 2 is t3, 3 is t4, 4 is t1-1, 5 is t2-1, 6 is t3-1, 7 is t4-1
    
    # t3, t4-1, t1-1, t2, t3-1, t4, t1, t2-1, t3, t4-1, t1-1, t2, t3-1, t4, t1, 
    # t2-1, t3, t4-1, t1-1, t2, t3-1, t4, t1, t2-1
    
    #t3, t4-1, t1-1, t2, t3-1, t4, t1, t2-1
    
    # [2, 7, 4, 1, 6, 3, 0, 5, 2, 7, 4, 1, 6, 3, 0, 5, 2, 7, 4, 1, 6, 3, 0, 5]
    newCombined = [2, 7, 4, 1, 6, 3, 0, 5]
    newCombined = [0, 5, 2, 7, 4, 1, 6, 3]
    print(f"8 gon identity commutator: {newCombined}")
    visualize_sequence(p, q, start_index=3, seq=newCombined, n_layers=n_layers, name="8 gon identity")


    p, q = 12, 12
    # 0 is t0, 1 is t1, 2 is t2... 6 is t0-1, 7 t1-1, 8 is t2-1... 11 is t5-1
    
    seq_a1_2 = [0, 5, 10]
    seq_b1_2 = [3, 8, 1]
    seq_c1_2 = [6, 11, 4]
    seq_d1_2 = [9, 2, 7]
    combined2 = compose_sequences(
        seq_a1_2, seq_b1_2, seq_c1_2, seq_d1_2
    )
    print(f"12 gon identity commutator {combined2}")
    visualize_sequence(p, q, start_index = 6, seq = combined2, n_layers = n_layers, name = "12 gon identity")
    
    # [0, 5, 10, 3, 8, 1, 6, 11, 4, 9, 2, 7]
    # t0, t5, t4-1, t3, t2-1, t1, t0-1, t5-1, t4, t3-1, t2, t1-1
    
    p, q = 16, 16
    
    seq_a1_3 = [0, 7, 14]
    seq_a2_3 = [5, 12, 3]
    seq_a3_3 = [10, 1, 8]
    seq_a4_3 = [15, 6, 13]
    seq_a5_3 = [4, 11, 2]
    seq_a6_3 = [9]
    
    # t0, t7, t6-1, t5, t4-1, t3, t2-1, t1, t0-1, t7-1, t6, t5-1, t4, t3-1, t2, t1-1 
    
    combined3 = compose_sequences(
        seq_a1_3, seq_a2_3, seq_a3_3, seq_a4_3, seq_a5_3, seq_a6_3
    )
    visualize_sequence(p, q, start_index = 5, seq = combined3, n_layers = n_layers, name = "16 gon identity")
    
    
    trustCombined = [0, 7, 14]
    
    
