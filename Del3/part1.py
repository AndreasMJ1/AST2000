from math import dist
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sp
import ast2000tools.constants as const
from tqdm import trange
from numba import jit
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission

seed = utils.get_seed('andrmj')
mission = SpaceMission(seed)
system = SolarSystem(seed)

ecc = system.eccentricities
aph_ang = system.aphelion_angles
m_ax = system.semi_major_axes
p_pos = system.initial_positions
p_vel = system.initial_velocities

#print(system.star_temperature)
#Lum = 4*np.pi*system.star_radii**2*const.sigma**4 
#print((system.star_radius*1e3)**2)
radius = 8961621.4961 
def flux_rec(dist):
    total_flux = 4*np.pi*((system.star_radius*1e3)**2)*const.sigma*system.star_temperature**4
    print(total_flux)
    fluxg = total_flux/(4*np.pi*dist**2)
    return fluxg

print(flux_rec(utils.AU_to_m(p_pos[0,0])))
#fr = flux_rec(p_pos[0,0])

#d_a = fr/(np.pi*radius**2)
#print(d_a)

#a = 40*0.12/4.7
#print(a)


