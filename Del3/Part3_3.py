#IKKE KODEMAL 
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

seed = 73494

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

Sm = system.star_mass
radius = system.radii[0]*1e3

def gravity(r):
    """
    Function for gravity, vectorized
    """
    rnorm = np.linalg.norm(r)
    f = ((G*homeplanet_mass)*r)/(rnorm**3)
    return f


def general_launch(phi,time,fuel_consume,Mass,F):
    """
    General launch function at any point and time in the simulated galaxy
    phi = the angle of the launch point (radians) relative to the equatorial line\\
    Time = the time of launch (years)\\
    fuel_consume = fuel consumption per second \\
    Mass = the current mass of the ship (kg)\\
    F = force of the thrusters (N)
    """
    dist = system.radii[0]*1e3
    phi = phi*np.pi/180
    rp = np.array((dist*np.cos(phi),dist*np.sin(phi)))
    
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

    explanpos = np.load("planet_trajectories.npz")
    npos = explanpos['planet_positions']
    plan_pos = np.einsum("ijk->kji",npos)
    
    velo = np.load('velocities.npy') 
    
    ind = int(time/0.0002)

    launch_point = plan_pos[ind,0,:] + utils.m_to_AU(rpi) 
    
    ycomp = np.cos(phi)*dist*2*np.pi/(utils.day_to_s(system.rotational_periods[0]))
    ypos = np.array((0,utils.m_to_AU(ycomp*timer)))
    pos_p = np.array((velo[ind,0,0]*utils.s_to_yr(timer),velo[ind,0,1]*utils.s_to_yr(timer)))
    fin_pos = plan_pos[ind,0,:] + utils.m_to_AU(rp) + ypos + pos_p
    
    return rp ,fin_pos, timer ,launch_point

if __name__ == '__main__': 
    explanpos = np.load("planet_trajectories.npz")
    npos = explanpos['planet_positions']
    plan_pos = np.einsum("ijk->kji",npos)
    times = explanpos['times'] 
    mean_force = 776938.7689392876
    rp, finpos, time ,launch_point = general_launch(180, times[1494] , 0.18871032743168 ,spacecraft_mass ,mean_force)