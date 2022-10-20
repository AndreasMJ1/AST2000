from tracemalloc import start
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

plan_vel = np.load('velocities.npy')
plan_pos = np.load('positions.npy')
def grav_star(r,m):
    r_abs = np.linalg.norm(r)
    sm = system.star_mass
    grav_force = -G*(m*sm)/(r_abs**3)*r
    return grav_force

def grav_planets(r,m,p):
    
    force = 0
    for l in range(n_planets):
        rv = r - plan_pos[p,l]
        r_abs = np.linalg.norm(rv)
        force += -G*(m*p_masses[l])*rv/(r_abs**3)
    return force 
    


def spacecraft_traj(init_t,init_r,init_v,time,dt):
    t_ind = int(init_t)
    #t_ind = 0
    mas = mission.spacecraft_mass*15
    
    mass = 150000          #Change value 
    print(mas,mass)
    steps = int(time/dt)
    int_vel = plan_vel[t_ind,0]
    #print(int_vel,init_v)
    v = np.zeros((steps+1,2))
    r = np.zeros((steps+1,2))
    r[0] = init_r + utils.km_to_AU(system.radii[0])
    v[0] = init_v + int_vel

    
    for i in trange(steps):
        force = grav_planets(r[i],mass,int(t_ind+i)) + grav_star(r[i],mass)

        a = force/mass
        if i ==0:
            print(v[i],v[i] + (a*dt/2),a*dt/2,a)
            print(dt)
        vh = v[i] + (a*dt/2)
        if i ==0:
            print(vh)
        r[i+1] = r[i] + vh*dt
        force = grav_planets(r[i+1],mass,t_ind+i+1) + grav_star(r[i+1],mass)
        a = force/mass
        v[i+1] = vh + a*dt/2

    
    return r,v

if __name__ =='__main__':
    start_pos = 956
    r0 = plan_pos[start_pos,0]
    r1 = m_ax[0]
    r2 = m_ax[2]
    ang = np.pi*(1-(1/(2*np.sqrt(2)))*np.sqrt(((r1/r2)+1)**3))
    #print(ang)
    ex = np.array((np.cos(ang)*r1,np.sin(ang)*r1))
    diff = np.zeros((4000,2))
    for i in range(4000):
        diff[i] = abs(np.sum(plan_pos[i,0] - ex))
        if  abs(np.sum(plan_pos[i,0] - ex)) == (6.5403251114593e-05):
            print(i)
    g = np.min(diff) #379
    mu = G*system.star_mass
    v1 = np.sqrt(mu/r2)*(1-np.sqrt((2*r1)/(r1+r2)))
    #print(v1)

    v2 = np.array((np.cos(ang)*v1,np.sin(ang)*v1))
    #print(v2)
    for i in range(4000):
        r1 = plan_pos[i,0] ; r2 = plan_pos[i,2]
        phi1 = (r1[1]/r1[0])
        phi2 = (r2[1]/r2[0])
        if abs(phi1 -phi2) < ang:
            #print(i, phi1,phi2) # 956 1494 2063
            break
    
    r,v= spacecraft_traj(start_pos,r0,v2,10,0.0002) # np.array((1.0760005429,0.31753332257920436))

    plt.plot(r[:,0],r[:,1])
    plt.plot(plan_pos[start_pos:50000,2,0],plan_pos[start_pos:50000,2,1])
    plt.plot(plan_pos[start_pos:50000,0,0],plan_pos[start_pos:50000,0,1])
    plt.axis('equal')
    plt.scatter(plan_pos[start_pos,0,0], plan_pos[start_pos,0,1])
    plt.scatter(plan_pos[start_pos,2,0], plan_pos[start_pos,2,1])
    plt.show()
    

