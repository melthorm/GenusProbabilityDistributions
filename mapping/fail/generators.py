import numpy as np
import matplotlib.pyplot as plt


class HyperbolicPolygon:
    def __init__(self, p: int, q: int):
        self.p = p
        self.q = q
        self.interior_points = []

        tan_pi_p = np.tan(np.pi / p)
        tan_pi2_minus_pi_q = np.tan(np.pi / 2 - np.pi / q)
        if tan_pi2_minus_pi_q + tan_pi_p <= 0:
            raise ValueError(f"(p,q)=({p},{q}) invalid for hyperbolic geometry")

        self.r = np.sqrt((tan_pi2_minus_pi_q - tan_pi_p) /
                         (tan_pi2_minus_pi_q + tan_pi_p))

        angles = np.linspace(0, 2*np.pi, p, endpoint=False)
        self.vertices = (self.r * np.exp(1j * angles)).tolist()
        self.side_pairs = self._compute_side_pairs()

    def _compute_side_pairs(self):
        pairs = []
        for i in range(self.p):
            z1 = self.vertices[i]
            z2 = self.vertices[(i+1) % self.p]
            w1 = self.vertices[(i-1) % self.p]
            w2 = self.vertices[i]
            pairs.append(((z1, z2), (w1, w2)))
        return pairs

    def reflect_point_across_line(self, p, a, b):
        v = p - a
        d = b - a
        u = d / abs(d)
        proj_scalar = np.real(v * np.conj(u))
        proj = proj_scalar * u
        return 2*(a + proj) - p

    def compute_circle_center(self, z1, z2):
        if np.allclose(z1, -z2):
            return None
        numerator = z1*(1 - abs(z2)**2) - z2*(1 - abs(z1)**2)
        denominator = np.conj(z1)*z2 - z1*np.conj(z2)
        if abs(denominator) < 1e-14:
            return None
        c = numerator / denominator
        c = self.reflect_point_across_line(c, z1, z2)
        return c

    def interpolate_along_arc(self, z1, z2, t=0.5):
        c = self.compute_circle_center(z1, z2)
        if c is None:
            return (1 - t)*z1 + t*z2
        r = abs(z1 - c)
        th1 = np.angle(z1 - c)
        th2 = np.angle(z2 - c)
        d = th2 - th1
        if d < -np.pi:
            d += 2*np.pi
        elif d > np.pi:
            d -= 2*np.pi
        thm = th1 + t*d
        return c + r*np.exp(1j*thm)

    def compute_interior_points(self):
        self.interior_points = []
        for (z1, z2), (w1, w2) in self.side_pairs:
            m = self.interpolate_along_arc(z1, z2)
            m2 = self.interpolate_along_arc(w1, w2)
            self.interior_points.append((m, m2))

    def visualize(self, transformed_polygons=None, n_points=100):
        plt.figure(figsize=(6,6))
        ax = plt.gca()
        circle = plt.Circle((0,0), 1, color='k', fill=False, lw=1)
        ax.add_artist(circle)

        def plot_polygon(verts, color='b', lw=2):
            N = len(verts)
            for i in range(N):
                z1 = verts[i]
                z2 = verts[(i+1)%N]
                c = self.compute_circle_center(z1, z2)
                if c is None:
                    x = [z1.real, z2.real]
                    y = [z1.imag, z2.imag]
                else:
                    r = abs(z1 - c)
                    th1 = np.angle(z1 - c)
                    th2 = np.angle(z2 - c)
                    d = th2 - th1
                    if d < -np.pi:
                        d += 2*np.pi
                    elif d > np.pi:
                        d -= 2*np.pi
                    thetas = th1 + np.linspace(0,1,n_points)*d
                    x = c.real + r*np.cos(thetas)
                    y = c.imag + r*np.sin(thetas)


                ax.plot(x, y, color=color, lw=lw)

        def plot_polygon_circle(verts, color='b', lw=2, n_points=500):
            N = len(verts)
            for i in range(N):
                z1 = verts[i]
                z2 = verts[(i+1)%N]
                c = self.compute_circle_center(z1, z2)
                if c is None:
                    # Straight line through origin
                    x = [z1.real, z2.real]
                    y = [z1.imag, z2.imag]
                else:
                    # Full circle orthogonal to unit circle
                    r = abs(z1 - c)
                    angles = np.linspace(0, 2*np.pi, n_points)
                    x = c.real + r * np.cos(angles)
                    y = c.imag + r * np.sin(angles)
                   

                ax.plot(x, y, color=color, lw=lw)


        plot_polygon(self.vertices, color='b', lw=5)
        if transformed_polygons:
            for poly in transformed_polygons:
                plot_polygon(poly, color='r', lw=1)

        ax.set_aspect('equal')
        ax.set_xlim(-1.05,1.05)
        ax.set_ylim(-1.05,1.05)


class HyperbolicGenerators:
    def __init__(self, polygon: HyperbolicPolygon):
        self.poly = polygon
        if not polygon.interior_points:
            polygon.compute_interior_points()
        self.generators = self._compute_all()

    @staticmethod
    def apply_matrix(z, M):
        a, b = M[0,0], M[0,1]
        c, d = M[1,0], M[1,1]
        return (a*z + b) / (c*z + d)

    def compute_one(self, side_index: int):
        (z1, z2), (w1, w2) = self.poly.side_pairs[side_index]
        m, m2 = self.poly.interior_points[side_index]

        f = lambda z: (z - z1) / (1 - np.conj(z1)*z)
        f_inv = lambda z: (z + z1) / (1 + np.conj(z1)*z)
        z2p = f(z2)

        g = lambda w: (w - w2) / (1 - np.conj(w2)*w)
        g_inv = lambda w: (w + w2) / (1 + np.conj(w2)*w)
        w1p = g(w1)

        alpha = w1p / z2p
        R = lambda z: alpha*z

        G = lambda z: g_inv(R(f(z)))

        a = 1
        b = G(0)*np.conj(a)
        norm = np.sqrt(np.abs(a)**2 - np.abs(b)**2)
        a /= norm
        b /= norm
        M = np.array([[a, b], [np.conj(b), np.conj(a)]], complex)

        s1 = np.cross([z2.real - z1.real, z2.imag - z1.imag],
                      [m.real - z1.real,   m.imag - z1.imag])
        s2 = np.cross([w1.real - w2.real, w1.imag - w2.imag],
                      [m2.real - w2.real, m2.imag - w2.imag])

        if np.sign(s1) != np.sign(s2):
            M = np.array([[np.conj(a), -b], [-np.conj(b), a]], complex)

        return M

    def _compute_all(self):
        out = []
        for i in range(len(self.poly.side_pairs)):
            out.append(self.compute_one(i))
        return out

    def apply(self, z, M):
        return self.apply_matrix(z, M)

    def apply_to_polygon(self, M):
        return [self.apply_matrix(z, M) for z in self.poly.vertices]

