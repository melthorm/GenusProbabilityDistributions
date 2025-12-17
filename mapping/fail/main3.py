import numpy as np
import matplotlib.pyplot as plt
from generators import HyperbolicPolygon, HyperbolicGenerators

def main(seq):
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
        print(verts)
        plot_poly(verts, colors[i % len(colors)])

    ax.set_aspect('equal')
    ax.set_xlim(-1.05,1.05)
    ax.set_ylim(-1.05,1.05)
    plt.show()

if __name__ == "__main__":
    main([0, 0, 0, 0, 0, 0])

