from ast import NotEq
from cmath import isclose
from inspect import isclass
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
from PartA import spacecraft_traj
seed = utils.get_seed('andrmj')

mission = SpaceMission(seed)
system = SolarSystem(seed)

ecc = system.eccentricities
aph_ang = system.aphelion_angles
m_ax = system.semi_major_axes
p_pos = system.initial_positions
p_vel = system.initial_velocities

plt.style.use('dark_background')
G = 4*np.pi**2
Sm = system.star_mass

def grav(r): 
    r_norm = np.linalg.norm(r)         #Function for gravity, vectorized
    a = -G*Sm*(r/(r_norm**3))
    return a 

v0 = np.array([[p_vel[0,0],p_vel[1,0]],[p_vel[0,1],p_vel[1,1]],[p_vel[0,2],p_vel[1,2]],[p_vel[0,3],p_vel[1,3]],
     [p_vel[0,4],p_vel[1,4]],[p_vel[0,5],p_vel[1,5]],[p_vel[0,6],p_vel[1,6]]]) #Init values

r0 = np.array([[p_pos[0,0],p_pos[1,0]],[p_pos[0,1],p_pos[1,1]],[p_pos[0,2],p_pos[1,2]],[p_pos[0,3],p_pos[1,3]],
           [p_pos[0,4],p_pos[1,4]],[p_pos[0,5],p_pos[1,5]],[p_pos[0,6],p_pos[1,6]]]) #Init values 

def sim_orbits(steps,dt):            #Simulation Loop
    r = np.zeros((steps,7,2))
    v = np.zeros((steps,7,2))
    v[0] = v0
    r[0] = r0
    cnt = 0
    closest = np.zeros((steps,2))
    for i in trange(steps-1):
        for p in range(7):
                    a = grav(r[i,p])
                    vh = v[i,p] + a*dt/2 
                    r[i+1,p] = r[i,p] + vh*dt 
                    a = grav(r[i+1,p])
                    v[i+1,p] = vh + a*dt/2                  
    return r , v 

color_list = ['Firebrick','Chartreuse','Khaki','Sienna','CornflowerBlue','Teal','Fuchsia']

r,v = sim_orbits(3126,0.0002)                         #Unpacking simulation 

if __name__ == '__main__':
    plan_pos = np.load('positions.npy')
    plt.scatter(0,0,color = 'black')                  #Plotting simulation 
    plt.plot(r[:,0,0],r[:,0,1])
    plt.plot(r[:,2,0],r[:,2,1])
    plt.legend(['0','2'])
    plt.xlabel('Distance (AU)')
    plt.ylabel('Distance (AU)')

    plt.scatter(r[3125,0,0],r[3125,0,1])
    plt.scatter(r[3125,2,0],r[3125,2,1])
    plt.plot((r[3125,0,0],r[3125,2,0]),(r[3125,0,1],r[3125,2,1]))
    plt.scatter(plan_pos[6125,2,0],plan_pos[6125,2,1])

    r0 = plan_pos[3125,0]
    r,v = spacecraft_traj(3125,r0,np.array((-3.5760005429,-6.31753332257920436)),0.6,0.0002)

    plt.plot(r[:,0],r[:,1])
    

    plt.show()
    

    planet_diff = abs(r[:,0] - r[:,2])
    ping = np.min(planet_diff)
    least = np.where(planet_diff == ping)



    #rshape = np.reshape(r,(2,7,119150))
    #mission.verify_planet_positions(119150*0.0002, rshape)
