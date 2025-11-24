import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

class HyperbolicPolygon:
    def __init__(self, vertices: list[complex], 
                 side_pairs: list[tuple[tuple[complex, complex], 
                 tuple[complex, complex]]]):
        self.vertices = vertices
        self.side_pairs = side_pairs
        self.interior_points = []  # list of tuples (m_i, m_i_prime)

    def reflect_point_across_line(self, p, a, b):
        """Reflect complex point p across the line through complex points a and b.
        Returns the reflected complex number. Assumes a != b."""
        v = p - a
        d = b - a
        if abs(d) == 0:
            raise ValueError("a and b must be distinct for line reflection")
        u = d / abs(d)                        # unit direction (complex)
        # projection scalar of v onto u (real)
        proj_scalar = np.real(v * np.conj(u))
        proj = proj_scalar * u
        # reflected point: a + proj - (v - proj) = 2*(a+proj) - p
        return 2*(a + proj) - p
        
    def compute_circle_center(self, z1, z2):
        """Return center of circle orthogonal to unit circle passing through z1,z2.
        Selects the branch outside the unit disk for inward-bending arcs."""
        if np.allclose(z1, -z2):
            return None  # straight line geodesic through origin

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
            return (1 - t)*z1 + t*z2  # straight line

        r = abs(z1 - c)

        theta1 = np.angle(z1 - c)
        theta2 = np.angle(z2 - c)

            

        # shortest angular path
        delta = theta2 - theta1
        if delta < -np.pi:
            delta += 2*np.pi
        elif delta > np.pi:
            delta -= 2*np.pi
        theta_m = theta1 + t*delta
        #print(f"center is {c}, z1 is {z1}, radius is {r}, theta is {theta1}, {theta2}, thetad is {theta_m}")

        return c + r * np.exp(1j * theta_m)


    def compute_interior_points(self):
        """
        Compute interior points (hyperbolic midpoints) for each side and its paired side.
        Stores them in self.interior_points as list of tuples (m_i, m_i_prime).
        """
        self.interior_points = []
        for side_pair in self.side_pairs:
            (z1, z2), (w1, w2) = side_pair
            m = self.interpolate_along_arc(z1, z2, t=0.5)
            m_prime = self.interpolate_along_arc(w1, w2, t=0.5)
            self.interior_points.append((m, m_prime))

    def compute_generator(self, side_index: int):
        """
        Compute the SU(1,1) Möbius generator for the side-pair using the three-point method.
        Maps z1 -> w1, z2 -> w2, m -> m_prime.
        Returns:
            M: 2x2 SU(1,1) matrix
            M_inv: inverse matrix
        """
        
        # Extract endpoints and interior points
        (z1, z2), (w2, w1) = self.side_pairs[side_index]
        m, m_prime = self.interior_points[side_index]

        print(f"Side index: {side_index}")
        print(f"Source endpoints z1,z2: {z1}, {z2}")
        print(f"Target endpoints w1,w2: {w1}, {w2}")
        print(f"Interior points m, m_prime: {m}, {m_prime}")

        # Step 1: map z1 -> 0
        f = lambda z: (z - z1) / (1 - np.conj(z1) * z)
        f_inv = lambda z: (z + z1) / (1 + np.conj(z1) * z)
        z2p = f(z2)
        mp = f(m)
        print(f"After mapping z1->{f(z1)}: z2 mapped = {z2p}, m mapped = {mp}")

        # Step 2: map w1 -> 0
        g = lambda w: (w - w1) / (1 - np.conj(w1) * w)
        g_inv = lambda w: (w + w1) / (1 + np.conj(w1) * w)
        w2p = g(w2)
        mp_prime = g(m_prime)
        print(f"After mapping w1->{g(w1)}: w2 mapped = {w2p}, m_prime mapped = {mp_prime}")

        # Step 3: rotation around origin
        alpha = w2p / z2p
        R = lambda z: alpha * z
        print(f"Rotation factor alpha: {alpha}")
        print(f"Check rotation: R(z2p) = {R(z2p)} should equal w2p = {w2p}")

        # Step 4: compose full Möbius map
        G = lambda z: g_inv(R(f(z)))
        mapped_m = G(m)
        print(f"Mapped interior point: {mapped_m} (should match m_prime: {m_prime})")

        # Step 5: extract SU(1,1) matrix
        # Solve G(0) = b / conj(a), choose a = 1
        a = 0.8
        b = G(0) * np.conj(a)
        norm = np.sqrt(np.abs(a)**2 - np.abs(b)**2)
        print(f"a = {a}, b = {b}, norm = {norm}")
        a /= norm
        b /= norm
        
        print(f"a = {a}, b = {b}, norm = {norm}")
        M = np.array([[a, b], [np.conj(b), np.conj(a)]], dtype=complex)
        M_inv = np.array([[np.conj(a), -b], [-np.conj(b), a]], dtype=complex)

        # Orientation check
        if np.sign(np.cross([z2.real - z1.real, z2.imag - z1.imag],
                            [m.real - z1.real, m.imag - z1.imag])) != \
           np.sign(np.cross([w2.real - w1.real, w2.imag - w1.imag],
                            [m_prime.real - w1.real, m_prime.imag - w1.imag])):
            print("Orientation flipped, swapping generator with inverse.")
            M, M_inv = M_inv, M

        print(f"Final generator matrix M:\n{M}")
        print(f"Inverse M_inv:\n{M_inv}")

        return M, M_inv




    
    def apply_generator(self, z: complex, generator_matrix):
        """
        Apply the Möbius map to a point in the disk.
        """
        a, b = generator_matrix[0,0], generator_matrix[0,1]
        c, d = generator_matrix[1,0], generator_matrix[1,1]
        return (a*z + b) / (c*z + d)

    def apply_generator_to_polygon(self, generator_matrix):
        """
        Apply a generator to all vertices of the polygon.
        Returns a list of transformed vertices.
        """
        return [self.apply_generator(z, generator_matrix) for z in self.vertices]

    def compute_all_generators(self):
        """
        Loop over all side-pairs and compute generators and inverses.
        Stores results in lists self.generators and self.generators_inv.
        """
        self.generators = []
        self.generators_inv = []
        # Make sure interior points are computed
        if not self.interior_points or len(self.interior_points) != len(self.side_pairs):
            self.compute_interior_points()
        for i in range(len(self.side_pairs)):
            M, M_inv = self.compute_generator(i)
            self.generators.append(M)
            self.generators_inv.append(M_inv)

    def visualize(self, transformed_polygons: list[list[complex]] = None, n_points=100):
        """
        Plots the polygon with hyperbolic geodesics (circle arcs orthogonal to unit circle).
        Optionally also plots transformed copies of the polygon.
        """

        plt.figure(figsize=(6,6))
        ax = plt.gca()
        
        # Draw unit circle
        circle = plt.Circle((0,0), 1, color='k', fill=False, lw=1)
        ax.add_artist(circle)
        
        def plot_polygon(vertices, color='b', lw=2):
            #print(vertices)
            # Draw each side as hyperbolic geodesic
            N = len(vertices)
            for i in range(N):
                z1 = vertices[i]
                z2 = vertices[(i+1)%N]
                # Compute circle center for inward-bending arc
                c = self.compute_circle_center(z1, z2)
                if c is None:
                    # straight line
                    x = [z1.real, z2.real]
                    y = [z1.imag, z2.imag]
                else:
                    r = abs(z1 - c)
                    theta1 = np.angle(z1 - c)
                    theta2 = np.angle(z2 - c)
                    # shortest angular path
                    delta = theta2 - theta1
                    if delta < -np.pi:
                        delta += 2*np.pi
                    elif delta > np.pi:
                        delta -= 2*np.pi
                    thetas = theta1 + np.linspace(0,1,n_points)*delta
                    x = c.real + r*np.cos(thetas)
                    y = c.imag + r*np.sin(thetas)
                #print(f"z is {z1} to {z2}, c is {c}, and points are {x}, {y}")
                ax.plot(x, y, color=color, lw=lw)

        # Plot fundamental polygon
        plot_polygon(self.vertices, color='b', lw=5)

        # Plot transformed polygons if given
        if transformed_polygons:
            for poly in transformed_polygons:

                plot_polygon(poly, color='r', lw=1)

        ax.set_aspect('equal')
        ax.set_xlim(-1.05,1.05)
        ax.set_ylim(-1.05,1.05)
        plt.show()

