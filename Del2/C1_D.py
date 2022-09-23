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


def gravS(r:np.asarray): 
    r_norm = np.linalg.norm(r)         #Function for gravity, vectorized
    a = G*Sm*p3*(r/(r_norm**3))
    return a 

def sim_orbit1(steps,dt):              #Simulation Loop
    cum_mass = Sm+p3 
    red_mass = Sm*p3/(cum_mass)
    sv = -(p3*p_vel[1,0])/Sm
    r = np.zeros((steps,2,2))
    v = np.zeros((steps,2,2))
    r0 =  np.array([[0,0],[p_pos[0,0],p_pos[1,0]]]) #np.array([[-red_mass*np.array([p_pos[0,0],p_pos[1,0]])/Sm],red_mass*np.array([p_pos[0,0],p_pos[1,0]])/p3]) 
    v0 = np.array([[0,sv],[p_vel[0,0],p_vel[1,0]]]) #np.array([[0,sv],[p_vel[0,0],p_vel[1,0]]]) #
    v[0] = v0
    r[0] = r0
    dt = dt
    
    for i in trange(steps-1):
        a_s = gravS(r[i,1]-r[i,0])/Sm
        a_p = gravS(r[i,1]-r[i,0])/p3

        vhs = v[i,0] + a_s*dt/2
        vhp = v[i,1] + a_p*dt/2
        

        r[i+1,0] = r[i,0] + vhs*dt
        r[i+1,1] = r[i,1] + vhp*dt 

        f_s = gravS(r[i+1,1]-r[i+1,0])
        a_s = -f_s/Sm

        f_p = gravS(r[i+1,1]-r[i+1,0])
        a_p = f_p/p3

        v[i+1,0] = vhs + a_s*dt/2
        v[i+1,1] = vhp + a_p*dt/2

    return r , v


r1,v1 = sim_orbit1(119000,0.0002)
plt.plot(r1[:,0,0],r1[:,0,1])
#plt.plot(r1[:,1,0],r1[:,1,1])
plt.show()
