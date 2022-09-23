from ast import NotEq
from lib2to3.pytree import type_repr
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
#from A1 import analytic_orbits
#from A2 import grav


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


def gravity(r,pmass):
    r_norm = np.linalg.norm(r)
    f = ((G*pmass*Sm)/(r_norm**3))*r 
    return f 


def solar_orbit(N,planetindx): #N = Antall steg, planetindx = liste med planetIDer
    dt = 0.0002 
    masses = np.zeros(len(planetindx))
    for i in range(len(masses)):
        masses[i] = planetindx[i]
    r = np.zeros((N,len(masses),2))
    v = np.zeros((N,len(masses),2))
    r0 = np.zeros((1,len(masses),2))
    v0 = np.zeros((1,len(masses),2))

    for i in range(len(masses)):
        index = int(masses[i])
        v0[0,i] = np.array([p_vel[0,index],p_vel[1,index]])
        r0[0,i] = np.array([p_pos[0,index],p_pos[1,index]])
    r[0] = r0
    v[0] = v0

    p_masses = np.zeros(len(masses))
    for i in range(len(masses)):
        ind = int(masses[i]) 
        p_masses[i] = system.masses[ind]
    t_red_mass = 1 ; b_red_mass = 0 
    sigma_p = 0
    a_s = np.array([0,0],dtype= 'float64')
    for i in range(len(masses)):
        sigma_p+= v[0,i]*p_masses[i]
        a_s = (gravity(r[0,i],p_masses[i])/Sm)
        t_red_mass *= p_masses[i]
        b_red_mass += p_masses[i]

    red_mass = Sm*t_red_mass/(b_red_mass+Sm)

    sun_v = np.zeros((N,2))
    sun_v[0] = -(sigma_p/Sm)
    
    sun_r = np.zeros((N,2))
    
    sunr0 = np.array([-np.sum(r0[0,:,0])*red_mass/Sm,-np.sum(r0[0,:,1])*red_mass/Sm])
    sun_r[0] = sunr0

    for s in trange(N-1):
        #a_s = np.array([0,0],dtype= 'float64') # np.array((1,2),dtype= 'float64')
        
        vhs = sun_v[s] + a_s*dt/2
        sun_r[s+1] = sun_r[s] +vhs*dt
        sun_v[s+1] = vhs + a_s*dt/2
        #print(a_s)
        if s == 0:
            pass
        elif p == len(masses)-1:
            a_s = np.array([0,0],dtype= 'float64')
        else:
            vhs = sun_v[s] + a_s*dt/2
            sun_r[s+1] = sun_r[s] +vhs*dt
            sun_v[s+1] = vhs + a_s*dt/2
            print(s, a_s)
        for p in range(len(masses)):
            a_s += (gravity(r[s,p]-sun_r[s],p_masses[p])/Sm)
            #print(a_s)
            a = (gravity(r[s,p],p_masses[p])/p_masses[p])
            vh = v[s,p] - a*dt/2
            r[s+1,p] = r[s,p] + vh*dt
            a = (gravity(r[s+1,p],p_masses[p])/p_masses[p])
            v[s+1,p] = vh - a*dt/2
    return sun_r, sun_v, r,v

sun_r, sun_v, r, v = solar_orbit(119000, np.array([0,1,2,3,4,5]))

plt.plot(sun_r[:,0],sun_r[:,1])
for i in range(len(r[0,:,0])):
    plt.plot(r[:,i,0],r[:,i,1])


plt.legend(['sun','0','2','5'])
plt.show()





    
    