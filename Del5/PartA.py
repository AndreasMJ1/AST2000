import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sp
import ast2000tools.constants as const
from tqdm import trange
from numba import njit
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
p_masses = system.masses
p_radii = system.radii
G = 4*np.pi**2
n_planets = len(p_masses)

def grav_star(r,m):
    r_abs = np.linalg.norm(r)
    sm = system.star_mass
    grav_force = -G*(m*sm)/(r_abs**3)*r
    return grav_force

def grav_planets(r,m,p):
    plan_pos = np.load('positions.npy')
    force = 0
    for i in range(n_planets):
        rv = r - plan_pos[p,i]
        r_abs = np.linalg.norm(rv)
        force += -G*(m*p_masses[i])*rv/(r_abs**3)
    return force 
    


def spacecraft_traj(init_t,init_r,init_v,time,dt):
    mass = 150000          #Change value 
    steps = int(time/dt)
    #t = np.linspace(init_t,init_t+time,steps)
    v = np.zeros((steps+1,2))
    r = np.zeros((steps+1,2))
    r[0] = init_r
    v[0] = init_v 
    t_ind = int(init_t*dt)

    for i in trange(steps):
        force = grav_planets(r[i],mass,t_ind+i) + grav_star(r[i],mass)
        a = force/mass
        vh = v[i] +a*dt/2
        r[i+1] = r[i] + vh*dt
        force = grav_planets(r[i+1],mass,t_ind+i) + grav_star(r[i+1],mass)
        a = force/mass
        v[i+1] = vh + a*dt/2
    
    return r,v

if __name__ =='__main__':

    plan_pos = np.load('positions.npy')
    #print(plan_pos)
    r0 = plan_pos[100,0]
    r,v = spacecraft_traj(100,r0,np.array((1.0760005429,0.31753332257920436)),0.2,0.0002)

    plt.plot(r[:,0],r[:,1])
    plt.show()

