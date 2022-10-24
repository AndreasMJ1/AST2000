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
from part3 import *
seed = utils.get_seed('andrmj')

mission = SpaceMission(seed)
system = SolarSystem(seed)

ecc = system.eccentricities
aph_ang = system.aphelion_angles
m_ax = system.semi_major_axes
p_pos = system.initial_positions
p_vel = system.initial_velocities
spacecraft_mass = mission.spacecraft_mass*15
homeplanet_dist = 1.2289822738 #AU
homeplanet_mass = 1.5616743232192E25 #7.80837162e-06 m_sun


G = 6.6743e-11 
#G = 4*np.pi**2
Sm = system.star_mass

def gravity(r):
    rnorm = np.linalg.norm(r)
    f = ((G*homeplanet_mass)*r)/(rnorm**3)
    return f


def general_launch(phi,time,fuel_consume,Mass,F):
    
    dist = 8.961621*1E6
    phi = phi*np.pi/180
    rp = np.array((dist*np.cos(phi),dist*np.sin(phi)))
    print(rp, 'start_pos')
    rpi = np.array((dist*np.cos(phi),dist*np.sin(phi)))
    dt = 0.001
    v = np.array((0,0))
    fc = 0
    timer = 0
    F = np.array((np.cos(phi)*F,np.sin(phi)*F))
    vesc = np.sqrt((G*2*homeplanet_mass)/(np.linalg.norm(rp)))
    while np.linalg.norm(v) <= vesc:
        fc += fuel_consume*dt  
        M = Mass - fc         
        a1 = F/M
        a2 = -gravity(rp)
        a = a1 + a2
        v = v + a*dt
        rp = rp + v*dt
        timer += dt 
    #key= np.load('planet_trajectories.npz')
    #plan_pos = key['planet_positions']
    plan_pos = np.load('positions.npy')
    
    velo = np.load('velocities.npy') 
    


    ind = int(time/0.0002)
    #print(ind)

    launch_point = plan_pos[ind,0,:] + utils.m_to_AU(rpi) #launch_point = plan_pos[:,0,ind] + utils.m_to_AU(rpi) 
    #print(launch_point)
    ycomp = np.cos(phi)*dist*2*np.pi/(utils.day_to_s(system.rotational_periods[0]))
    ypos = np.array((0,utils.m_to_AU(ycomp*timer)))
    pos_p = np.array((velo[ind,0,0]*utils.s_to_yr(timer),velo[ind,0,1]*utils.s_to_yr(timer)))
    fin_pos = plan_pos[ind,0,:] + utils.m_to_AU(rp) + ypos + pos_p
    #print(velo[ind])
    print(np.array([-9.28709058, -0.13534456])- velo[ind,0])
    return rp ,fin_pos, timer ,launch_point


if __name__ == '__main__': #1493*0.0002
    mean_force = 776938.7689392876
    rp, finpos, time ,launch_point = general_launch(180, 1493*0.0002 , fuel_consume ,spacecraft_mass ,mean_force)
    print(rp,finpos,time,launch_point)
    #k = rpi - utils.m_to_AU(rp)
    exac_pos = [8.69554508e-04 ,1.22575863e+00]
    mission.set_launch_parameters(mean_force, fuel_consume, spacecraft_mass, 500, launch_point , 1493*0.0002)
    mission.launch_rocket(0.001)
    mission.verify_launch_result(exac_pos) #[1.22905961e+00 8.38221473e-05]
    #mission.take_picture()
    

    #mission.verify_manual_orientation(exac_pos, [-9.28709058, -0.13534456], 192) # 192 


