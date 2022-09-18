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

p3 = system.masses[0]

G = 4*np.pi**2
Sm = system.star_mass
def gravS(r:np.ndarray): 
    r_norm = np.linalg.norm(r)         #Function for gravity, vectorized
    a = -G*p3*(r/(r_norm**3))
    return a 


def sim_orbit1(steps,dt):            #Simulation Loop
    r = np.zeros((steps,2,2))
    v = np.zeros((steps,2,2))
    r0 = np.array([[0,0],[p_pos[0,0],p_pos[1,0]]])
    v0 = np.array([[0,0],[p_vel[0,0],p_vel[1,0]]])
    v[0] = v0
    r[0] = r0
    dt = dt
    mc = (Sm*system.masses[0])/(Sm+system.masses[0])
    for i in trange(steps-1):
        #sim p
        a = grav((r[i,1]-r[i,0]))
        vh = v[i,1] + a*dt/2 
        r[i+1,1] = r[i,1] + vh*dt 
        a = grav(r[i+1,1])
        v[i+1,1] = vh + a*dt/2
        #sim s 
        a1 = gravS((r[i,0]-r[i,1]))
        vh1 = v[i,0] + a1*dt/2 
        r[i+1,0] = r[i,0] + vh1*dt 
        a1 = gravS(r[i+1,0])
        v[i+1,0] = vh1 + a1*dt/2
    return r , v 

#print(g)
#print(gravS(np.array([p_pos[0,3],p_pos[1,3]])))

r1,v1 = sim_orbit1(11900,0.002)

#plt.plot(r1[:,0,0],r1[:,0,1])
plt.plot(r1[:,1,0],r1[:,1,1])

plt.show()

#r,v,a_chk,cnt,push_p = sim_orbits(11900,0.002)
#plt.plot(r[:,3,0],r[:,3,1])
#plt.scatter(0,0)
#plt.show()