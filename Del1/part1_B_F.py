#IKKE KODEMAL 

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sp
import ast2000tools.constants as const
from tqdm import trange
from numba import njit
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission


k = 1.38064852e-23 #Boltzmann's Constant
T = 3e3           #Temperature in Kelvin
L = 10e-6         #Box Length in meters
N = 1e5           #Number of particles
m = 3.3474472e-27 #Mass of individual particle in kg
G = 6.6743e-11    #Gravitational Constant 
seed = 73494
mission = SpaceMission(seed)
system = SolarSystem(seed)
spacecraft_mass = mission.spacecraft_mass*15
homeplanet_dist = 1.2289822738 #AU
homeplanet_mass = 1.5616743232192E25 #7.80837162e-06 m_sun
radius = system.radii[0]

pos1 = [homeplanet_dist+utils.km_to_AU(radius),0]
vel = []
time1 = []

pos0 = np.array([homeplanet_dist+utils.km_to_AU(radius),0])

#INITIAL CONDITIONS
sigma =  np.sqrt((k*T)/m)    #The standard deviation of our particle velocities
 
x =  np.random.uniform(0,L, size = (int(N), 3))             # Position vector
v =  np.random.normal(0,sigma, size = (int(N), 3))          # Velocity vector

#SIMULATION VARIABLES
r = 10e-9               #Simulation Runtime in Seconds
steps = 1000            #Number of Steps Taken in Simulation
dt = r/steps            #Simulation Step Length in Seconds
exiting = 0             #The total number of particles that have exited the gas box
f = 0                   #Used to calculate Force/Second later in the simulation
l = 0                   #Amount That has bounced inside the box 
s = L/4                 #Length from edge to escape hole (engine)

#RUNNING THE CONDITIONAL INTEGRATION LOOP
@njit()
def particle_sim(x,v,l,exiting,f):
    """
    Simulating the performance of a singular engine box
    """
    for i in range(int(steps)):
        x += dt*v    
        for j in range(int(N)):
            if s <= x[j][0] <= 3*s and s <= x[j][1] <= 3*s and x[j][2] <= 0:
                exiting+=1
                f += v[j][2]*m/dt
                x[j] =  np.random.uniform(0,L, size = (1, 3))         # Position vector, fill in uniform distribution in all 3 dimensions
                v[j] =  np.random.normal(0,sigma, size = (1, 3)) 
            else:
                for u in range(3):
                    if x[j][u] >= L or x[j][u] <=0:
                        v[j][u] = -v[j][u]
                        l += 1  
    return x, l, exiting ,f

def rocketengine_perf(mean_force):
    """
    Calculating fuel consumption
    """
    guess_boxes = 1.6e13
    fuel_consume = box_mass * guess_boxes      #Consume per second 
    return fuel_consume


x,l,exiting,tf = particle_sim(x,v,l,exiting,f) #Unpacking values from engine simulation

particles_per_second = exiting/r              
mean_force = -tf                              
box_mass = particles_per_second*m                     
fuel_consume = rocketengine_perf(mean_force)



def gravity(r):
    """
    Funcion for gravity at distance r 
    """
    f = -(G*homeplanet_mass)/(r**2)
    return f

def orbit_launch(F,Mass,fuel):
    """
    Numerically simualting the rocket launch
    """
    vesc = np.sqrt((G*2*homeplanet_mass)/((radius*1e3)))
    dist = (radius*1e3)
    dt = 0.001
    v=0
    fc = 0
    timer = 0
    pos= []
    while v <= vesc:
        
        fc += fuel_consume*dt  
        
        M = Mass - fc         
        a = F/M + gravity(dist)
        v = v + a*dt
        dist += v*dt
        pos.append(dist)
        timer += dt
        vesc = np.sqrt((G*2*homeplanet_mass)/(dist))
        if v < 0:
            break
    return v , timer, pos ,dist

if __name__ == '__main__':

    mission.set_launch_parameters(mean_force*1.6e13, fuel_consume, spacecraft_mass, 500, pos0, 0)
    mission.launch_rocket(0.001)

    vy0 = utils.AU_pr_yr_to_m_pr_s(system.initial_velocities[1,0]) 

    rotv = 2*np.pi*radius*1e3/(utils.day_to_s(system.rotational_periods[0])) #Rotanional velocity
    vel, time, pos ,dist= orbit_launch(mean_force*1.6e13,spacecraft_mass,fuel_consume)
    
    x1 = dist
    x = utils.m_to_AU(x1)  + pos0[0]
    y = (vy0 +rotv) *(401.44)
    y1 = utils.m_to_AU(y)

    position = np.array([x,y1]) #Calculated position error of 5.8951e-05 AU 
    
    exac_pos = [1.22905961e+00, 8.38221473e-05] #exact position obtained from AST - shortcut

    mission.verify_launch_result(position)


    









    
