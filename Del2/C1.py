from ast import NotEq
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
from A1 import analytic_orbits
from A2 import grav


seed = utils.get_seed('andrmj')
mission = SpaceMission(seed)
system = SolarSystem(seed)

ecc = system.eccentricities
aph_ang = system.aphelion_angles
m_ax = system.semi_major_axes
p_pos = system.initial_positions
p_vel = system.initial_velocities

p3 = system.masses[3]

G = 4*np.pi**2
Sm = system.star_mass
def gravS(r:np.ndarray): 
    r_norm = np.linalg.norm(r)         #Function for gravity, vectorized
    a = G*p3*(r/(r_norm**3))
    return a 

g = grav(np.array([p_pos[0,3],p_pos[1,3]]))
print(g)
print(gravS(np.array([p_pos[0,3],p_pos[1,3]])))


#r,v,a_chk,cnt,push_p = sim_orbits(11900,0.002)
#plt.plot(r[:,3,0],r[:,3,1])
#plt.scatter(0,0)
#plt.show()