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
seed = utils.get_seed('andrmj')
mission = SpaceMission(seed)
system = SolarSystem(seed)

ecc = system.eccentricities
aph_ang = system.aphelion_angles
m_ax = system.semi_major_axes
p_pos = system.initial_positions
p_vel = system.initial_velocities


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
    dt = dt
    a_chk = []
    cnt = 0
    push_p = {0:[],1:[],2:[],3:[],4:[],5:[],6:[]}
    for i in trange(steps-1):
        for p in range(7):
            if i == 0 or i == int(steps/4):
                a = grav(r[i,p])
                vh = v[i,p] + a*dt/2 
                r[i+1,p] = r[i,p] + vh*dt 
                a = grav(r[i+1,p])
                v[i+1,p] = vh + a*dt/2
                areal = (np.linalg.norm((r[i,p,:]+r[i+1,p,:])/2) * np.linalg.norm((r[i+1,p,:]-r[i,p,:])))/2
                dist = np.linalg.norm((r[i+1,p,:]-r[i,p,:]))
                vel = dist/dt
                a_chk.append([areal,dist,vel])

            else: 
                a = grav(r[i,p])
                vh = v[i,p] + a*dt/2 
                r[i+1,p] = r[i,p] + vh*dt 
                a = grav(r[i+1,p])
                v[i+1,p] = vh + a*dt/2
                if np.sign(r[i,p,1]) != np.sign(r[i+1,p,1]):
                    cnt +=1
                    push_p[p].append(i)
        
    return r , v , a_chk, cnt,push_p

color_list = ['Firebrick','Chartreuse','Khaki','Sienna','CornflowerBlue','Teal','Fuchsia']

r,v,a_chk,cnt,push_p = sim_orbits(119150,0.0002)  #Unpacking simulation 

if __name__ == '__main__':
        
    for i in range(7):                                #Plotting Exact solution
        plt.plot(analytic_orbits(m_ax[i],ecc[i],aph_ang[i],119150,p_pos[0][i],p_pos[1][i])[0],analytic_orbits(m_ax[i],ecc[i],aph_ang[i]
        ,119150,p_pos[0][i],p_pos[1][i])[1],linestyle='dotted',color = f'{color_list[i]}')

    print((a_chk[0][0],a_chk[7][0],a_chk[0][0]-a_chk[7][0],a_chk[0][1],a_chk[7][1],a_chk[0][2],a_chk[7][2])) #(-8.282099940755405e-12, 0.0012898681485006699, 0.0012898732900769251, 6.449340742503349, 6.449366450384625)


    print(f"Kepler {119150*0.0002/20,np.sqrt(m_ax[0]**3)}")
    print(f"Netwon {119150*0.0002/20,np.sqrt((4*np.pi**2)/(4*np.pi**2*(system.star_mass+system.masses[0]))*m_ax[0]**3),}") #595*0.002

    print(system.types)

    plt.scatter(0,0,color = 'black')                  #Plotting simulation 
    for i in range(7):
        plt.plot(r[:,i,0],r[:,i,1],color = f'{color_list[i]}')
    plt.legend(['0','1','2','3','4','5','6'])
    plt.xlabel('Distance (AU)')
    plt.ylabel('Distance (AU)')
    plt.title('Comparing analytical and numeric calculations')
    plt.show()

    rshape = np.reshape(r,(2,7,119150))
    mission.verify_planet_positions(119150*0.0002, rshape)