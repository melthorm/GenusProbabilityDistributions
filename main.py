import NormalTorusSampler
import GenusNSampler
import TorusNoIntersectSampler
import numpy as np
import torch
import sys



""" NORMAL TORUS """

sampler = NormalTorusSampler.NormalTorusSampler(R=2.0, r=0.7, 
                                                mu1=0.0, kappa1=4.0, 
                                                mu2=1.57, kappa2=10.0, 
                                                device = "cpu")

# Plot points + actual torus surface
points = sampler.plot_samples(n_samples=2000, s=5, show_surface=True, return_points=True)


""" WEIRD AAH TORUS """
"""
sampler = NormalTorusSampler.NormalTorusSampler(    R=2.0, r=0.7,                 # Torus radii
    mu1=0, kappa1=2.0,          # von Mises for theta (minor circle)
    mu2=np.pi, kappa2=2.0,     # von Mises for phi (major circle)

                                                device = "cpu")

# Plot points + actual torus surface
points = sampler.plot_samples(n_samples=2000, s=5, show_surface=True, return_points=True)
"""

""" Example with uniform distribution """
torusIntersectUniform = TorusNoIntersectSampler.TorusNoIntersectSampler(
    R=2.0, r=0.7,                 # Torus radii
    mu1=0.0, kappa1=0.1,          # von Mises for theta (minor circle)
    mu2=0.0, kappa2=0.1,     # von Mises for phi (major circle)
    x_offset = 4.5
)


# Plot sampled points, torus surface, and the removed patch
torusIntersectUniform.plot_points(
    n_samples=2000, 
    s=3, 
    show_surface=True,
    show_second_torus = True,
    return_points=False
)

torusIntersectConcentrate = TorusNoIntersectSampler.TorusNoIntersectSampler(
                                            R=2.0, r=0.7, 
                                                mu1=0.0, kappa1=4.0, 
                                                mu2=0.5, kappa2=10.0, 
                                                device = "cpu",
    x_offset = 4.5
)


# Plot sampled points, torus surface, and the removed patch
torusIntersectConcentrate.plot_points(
    n_samples=2000, 
    s=3, 
    show_surface=True,
    show_second_torus = True,
    return_points=False
)


n = 4
x_spacing = 4.0

# Only first and second, last and second-last tori have intersections removed
intersect_map = [
    [4.0],      # torus 0 avoids intersecting torus 1
    [-4.0, 4.0], # torus 1 avoids torus 0 and torus 2
    [-4.0, 4.0],# torus 2 avoids torus 1 and torus 3
    [-4.0]       # torus 3 avoids torus 2
]

sampler = GenusNSampler.GenusNSampler(
    n=n,
    R=2.0,
    r=0.7,
    x_spacing=x_spacing,
    sigma=0.2,
    intersect_map=intersect_map
)

sampler.plot(n_samples=3000, s=5)

