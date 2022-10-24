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
    ang = np.arctan(init_r[1]/init_r[0])
    mass = 150000          #Change value 
    print(mas,mass)
    steps = int(time/dt)
    int_vel = plan_vel[t_ind,0]
    #print(int_vel,init_v)
    v = np.zeros((steps+1,2))
    r = np.zeros((steps+1,2))
    
    r[0] = init_r + np.array((utils.km_to_AU(100*system.radii[0])*np.cos(ang),100*utils.km_to_AU(system.radii[0])*np.sin(ang))) #np.array(utils.km_to_AU(system.radii[0])*np.cos(ang),np.sin(ang)*utils.km_to_AU(system.radii[0])) 
    v[0] = init_v + int_vel
    print(r[0])
    
    print(ang)

    for i in trange(steps):
        forcep = grav_planets(r[i],mass,int(t_ind+i))
        forces = grav_star(r[i],mass)
        force = forcep + forces
        a = force/mass
        
        vh = v[i] + (a*dt/2)

        r[i+1] = r[i] + vh*dt
        force = grav_planets(r[i+1],mass,t_ind+i+1) + grav_star(r[i+1],mass)
        a = force/mass
        v[i+1] = vh + a*dt/2

    
    return r,v

if __name__ =='__main__':
    start_pos = 1494   #956+400
    r0 = plan_pos[start_pos,0]
    r1 = m_ax[0]
    r2 = m_ax[2]
    ang = np.pi*(1-(1/(2*np.sqrt(2)))*np.sqrt(((r1/r2)+1)**3))
    ang1 = np.arctan(r0[1]/r0[0])
    #print(ang)
    ex = np.array((np.cos(ang)*r1,np.sin(ang)*r1))
    diff = np.zeros((4000,2))
    for i in range(4000):
        diff[i] = abs(np.sum(plan_pos[i,0] - ex))
        if  abs(np.sum(plan_pos[i,0] - ex)) == (6.5403251114593e-05):
            print(i)
    g = np.min(diff) #379
    mu = G*system.star_mass
    v1 = np.sqrt(mu/r1)*(np.sqrt((2*r2)/(r1+r2))-1)
    #print(v1)

    v2 = np.array((np.sin(ang1)*v1,np.cos(ang1)*v1))
    #print(v2)
    for i in range(4000):
        r1 = plan_pos[i,0] ; r2 = plan_pos[i,2]
        phi1 = (r1[1]/r1[0])
        phi2 = (r2[1]/r2[0])
        if abs(phi1 -phi2) < ang:
            #print(i, phi1,phi2) # 956 1494 2063
            break
    print(v2) 
    v2 = v2 * np.array((1,-1)) 
    v2 = v2 *1.07
    
    r,v= spacecraft_traj(start_pos,r0, v2,4,0.0002) # np.array((1.0760005429,0.31753332257920436)) 5.6*v2
    ang = np.arctan(r0[1]/r0[0])
    print(ang,'angle')
    print(np.sin(ang))
    plt.plot(r[:,0],r[:,1],color = 'Black')
    plt.plot(plan_pos[start_pos:20000,2,0],plan_pos[start_pos:20000,2,1])
    plt.plot(plan_pos[start_pos:20000,0,0],plan_pos[start_pos:20000,0,1])
    plt.axis('equal')
    plt.scatter(plan_pos[start_pos,0,0], plan_pos[start_pos,0,1])
    plt.scatter(plan_pos[start_pos,2,0], plan_pos[start_pos,2,1])
    
    star_mass = system.star_mass
    pos_diff = np.zeros((10000-start_pos))
    for i in trange(int(10000-start_pos)):
        
        pos_diff[i] = np.linalg.norm(plan_pos[start_pos+i,2] - r[i])
        if np.linalg.norm(pos_diff[i]) <= np.linalg.norm(r[i])*np.sqrt(p_masses[2]/(10*star_mass)):
            print(i)
            #plt.scatter(plan_pos[i+start_pos,2,0],plan_pos[i+start_pos,2,1])
            #plt.scatter(r[i,0],r[i,1])
    k = (np.min(pos_diff))
    #print(pos_diff)
    u =np.delete(pos_diff,[-1])
    #print(u[-1])
    #print(k)
    
    l = (np.where(pos_diff == k ))[0][0] 

    print(np.linalg.norm(r[l])*np.sqrt(p_masses[2]/(10*star_mass)))
    print(pos_diff[l])

    plt.scatter(plan_pos[int(l+start_pos),2,0],plan_pos[int(l+start_pos),2,1])
    plt.scatter(r[l,0],r[l,1])
            

    plt.show()

    

