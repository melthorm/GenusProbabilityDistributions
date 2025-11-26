import numpy as np
import matplotlib.pyplot as plt

class HyperbolicPolygon:
    def __init__(self, p: int, q: int):
        """
        Create a regular hyperbolic p-gon for a (p,q) tiling.
        Vertices are placed inside the Poincaré disk with correct hyperbolic radius.
        Side pairs are computed according to q, so that q polygons meet at each vertex.
        """
        self.p = p
        self.q = q
        self.interior_points = []

        # Compute Euclidean radius in Poincaré disk using correct hyperbolic formula
        tan_pi_p = np.tan(np.pi / p)
        tan_pi2_minus_pi_q = np.tan(np.pi / 2 - np.pi / q)
        if tan_pi2_minus_pi_q + tan_pi_p <= 0:
            raise ValueError(f"(p,q)=({p},{q}) does not give a hyperbolic polygon")
        self.r = np.sqrt((tan_pi2_minus_pi_q - tan_pi_p) / (tan_pi2_minus_pi_q + tan_pi_p))

        # Place vertices evenly on circle
        angles = np.linspace(0, 2*np.pi, p, endpoint=False)
        self.vertices = (self.r * np.exp(1j * angles)).tolist()

        # Compute side-pairs according to q
        self.side_pairs = self._compute_side_pairs()

    def _compute_side_pairs(self):
        """
        Compute the side pairs for a (p,q) tiling so that each edge maps to its
        adjacent polygon under the Möbius generators. 
        Returns a list of tuples: ((z1,z2), (w1,w2)).
        """
        pairs = []
        # Reflect each edge across the polygon center according to q
        for i in range(self.p):
            z1 = self.vertices[i]
            z2 = self.vertices[(i+1) % self.p]
            w1 = self.vertices[(i - 1) % self.p]  # q adjacency reflected
            w2 = self.vertices[i]
            pairs.append(((z1, z2), (w1, w2)))
        return pairs

    def reflect_point_across_line(self, p, a, b):
        v = p - a
        d = b - a
        if abs(d) == 0:
            raise ValueError("a and b must be distinct for line reflection")
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
            return (1 - t) * z1 + t * z2
        r = abs(z1 - c)
        theta1 = np.angle(z1 - c)
        theta2 = np.angle(z2 - c)
        delta = theta2 - theta1
        if delta < -np.pi:
            delta += 2*np.pi
        elif delta > np.pi:
            delta -= 2*np.pi
        theta_m = theta1 + t*delta
        return c + r * np.exp(1j * theta_m)

    def compute_interior_points(self):
        self.interior_points = []
        for (z1, z2), (w1, w2) in self.side_pairs:
            m = self.interpolate_along_arc(z1, z2)
            m_prime = self.interpolate_along_arc(w1, w2)
            self.interior_points.append((m, m_prime))

    def compute_generator(self, side_index: int):
        (z1, z2), (w1, w2) = self.side_pairs[side_index]
        m, m_prime = self.interior_points[side_index]

        f = lambda z: (z - z1) / (1 - np.conj(z1) * z)
        f_inv = lambda z: (z + z1) / (1 + np.conj(z1) * z)
        z2p = f(z2)
        mp = f(m)

        g = lambda w: (w - w2) / (1 - np.conj(w2) * w)
        g_inv = lambda w: (w + w2) / (1 + np.conj(w2) * w)
        w1p = g(w1)
        mp_prime = g(m_prime)

        alpha = w1p / z2p
        R = lambda z: alpha * z

        G = lambda z: g_inv(R(f(z)))

        a = 0.8
        b = G(0) * np.conj(a)
        norm = np.sqrt(np.abs(a)**2 - np.abs(b)**2)
        a /= norm
        b /= norm

        M = np.array([[a, b], [np.conj(b), np.conj(a)]], dtype=complex)
        M_inv = np.array([[np.conj(a), -b], [-np.conj(b), a]], dtype=complex)

        if np.sign(np.cross([z2.real - z1.real, z2.imag - z1.imag],
                            [m.real - z1.real, m.imag - z1.imag])) != \
           np.sign(np.cross([w1.real - w2.real, w1.imag - w2.imag],
                            [m_prime.real - w2.real, m_prime.imag - w2.imag])):
            M, M_inv = M_inv, M

        return M, M_inv

    def apply_generator(self, z: complex, generator_matrix):
        a, b = generator_matrix[0,0], generator_matrix[0,1]
        c, d = generator_matrix[1,0], generator_matrix[1,1]
        return (a*z + b) / (c*z + d)

    def apply_generator_to_polygon(self, generator_matrix):
        return [self.apply_generator(z, generator_matrix) for z in self.vertices]

    def compute_all_generators(self):
        self.generators = []
        self.generators_inv = []
        if not self.interior_points or len(self.interior_points) != len(self.side_pairs):
            self.compute_interior_points()
        for i in range(len(self.side_pairs)):
            M, M_inv = self.compute_generator(i)
            self.generators.append(M)
            self.generators_inv.append(M_inv)

    def visualize(self, transformed_polygons=None, n_points=100):
        plt.figure(figsize=(6,6))
        ax = plt.gca()
        circle = plt.Circle((0,0), 1, color='k', fill=False, lw=1)
        ax.add_artist(circle)

        def plot_polygon(vertices, color='b', lw=2):
            N = len(vertices)
            for i in range(N):
                z1 = vertices[i]
                z2 = vertices[(i+1)%N]
                c = self.compute_circle_center(z1, z2)
                if c is None:
                    x = [z1.real, z2.real]
                    y = [z1.imag, z2.imag]
                else:
                    r = abs(z1 - c)
                    theta1 = np.angle(z1 - c)
                    theta2 = np.angle(z2 - c)
                    delta = theta2 - theta1
                    if delta < -np.pi:
                        delta += 2*np.pi
                    elif delta > np.pi:
                        delta -= 2*np.pi
                    thetas = theta1 + np.linspace(0,1,n_points)*delta
                    x = c.real + r*np.cos(thetas)
                    y = c.imag + r*np.sin(thetas)
                ax.plot(x, y, color=color, lw=lw)

        plot_polygon(self.vertices, color='b', lw=5)
        if transformed_polygons:
            for poly in transformed_polygons:
                plot_polygon(poly, color='r', lw=1)
        ax.set_aspect('equal')
        ax.set_xlim(-1.05,1.05)
        ax.set_ylim(-1.05,1.05)

