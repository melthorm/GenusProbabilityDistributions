import numpy as np

# ------------------- Polygon & hyperbolic geometry -------------------
class HyperbolicPolygon:
    def __init__(self, p, q):
        self.p = p
        self.q = q
        self.vertices = self._compute_vertices()
        # pair sides for standard presentation: naive pairing
        self.side_pairs = [(i, (i + p//2) % p) for i in range(p//2)]

    def _compute_vertices(self):
        # Hyperbolic radius from origin for regular (p,q) polygon
        cosh_r = np.cos(np.pi/self.q)/np.sin(np.pi/self.p)
        if cosh_r <= 1:
            raise ValueError(f"(p,q)=({self.p},{self.q}) not hyperbolic")
        r = np.tanh(np.arccosh(cosh_r)/2)
        return [r * np.exp(2j*np.pi*i/self.p) for i in range(self.p)]

# ------------------- SU(1,1) generators -------------------
class HyperbolicGenerators:
    def __init__(self, poly: HyperbolicPolygon):
        self.poly = poly
        self.generators = self._compute_generators()

    def _compute_generators(self):
        gens = []
        for i, j in self.poly.side_pairs:
            z1, z2 = self.poly.vertices[i], self.poly.vertices[j]
            # SU(1,1) map sending z1 -> z2
            a = (1 + z2*np.conj(z1))/np.sqrt((1 - abs(z1)**2)*(1 - abs(z2)**2))
            b = (z2 - z1)/np.sqrt((1 - abs(z1)**2)*(1 - abs(z2)**2))
            norm = np.sqrt(abs(a)**2 - abs(b)**2)
            a /= norm
            b /= norm
            M = np.array([[a,b],[np.conj(b),np.conj(a)]], dtype=complex)
            gens.append(M)
        return gens

    @staticmethod
    def apply(z, M):
        a,b = M[0,0], M[0,1]
        c,d = M[1,0], M[1,1]
        return (a*z + b)/(c*z + d)

    def apply_sequence(self, z, seq):
        for idx in seq:
            if idx >= 0:
                M = self.generators[idx]
            else:
                M = np.linalg.inv(self.generators[-idx])
            z = self.apply(z, M)
        return z

    def check_commutator_identity(self, seq):
        deviations = []
        for vi,z in enumerate(self.poly.vertices):
            z_end = self.apply_sequence(z, seq)
            deviations.append(abs(z_end - z))
        return deviations

# ------------------- Run -------------------
if __name__ == "__main__":
    p, q = 8, 8
    poly = HyperbolicPolygon(p,q)
    gens = HyperbolicGenerators(poly)

    # Commutator sequence a1 b1 a1^-1 b1^-1 a2 b2 a2^-1 b2^-1
    seq = [0,1,-0,-1,2,3,-2,-3]

    deviations = gens.check_commutator_identity(seq)
    for vi, dev in enumerate(deviations):
        print(f"Vertex {vi}: deviation from identity = {dev}")

