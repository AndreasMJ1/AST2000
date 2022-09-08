import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sp
import ast2000tools.constants as const
from tqdm import trange
from numba import jit
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission


#CONSTANTS
k = 1.38064852e-23 #Boltzmann's Constant

#PHYSICAL VARIABLES
T = 3e3           #Temperature in Kelvin
L = 10e-6         #Box Length in meters
N = 1e4           #Number of particles
m = 3.3474472e-27 #Mass of individual particle in kg
G = 6.6743e-11    #Gravitational Constant 
seed = utils.get_seed('andrmj')
mission = SpaceMission(seed)
system = SolarSystem(seed)
spacecraft_mass = mission.spacecraft_mass + mission.spacecraft_mass*20
homeplanet_dist = 1.2289822738 #AU
homeplanet_mass = 1.5616743232192E25 #7.80837162e-06 m_sun
pos = [homeplanet_dist+utils.km_to_AU(8961.62),]
vel = []
time1 = []
"""
My mission starts on planet 0, which has a radius of 8961.62 kilometers.
My system has a 1.29825 solar mass star with a radius of 903916 kilometers.
Planet 0 is a rock planet with a semi-major axis of 1.22577 AU.
Planet 1 is a rock planet with a semi-major axis of 1.58381 AU.
Planet 2 is a rock planet with a semi-major axis of 2.62217 AU.
Planet 3 is a gas planet with a semi-major axis of 10.9258 AU.
Planet 4 is a gas planet with a semi-major axis of 7.07649 AU.
Planet 5 is a gas planet with a semi-major axis of 5.07624 AU.
Planet 6 is a rock planet with a semi-major axis of 3.70428 AU.
My spacecraft has a mass of 1100 kg and a cross-sectional area of 16 m^2.
"""


#INITIAL CONDITIONS
sigma =  np.sqrt((k*T)/m)    #The standard deviation of our particle velocities
 
x =  np.random.uniform(0,L, size = (int(N), 3))             # Position vector
v =  np.random.normal(0,sigma, size = (int(N), 3))          # Velocity vector

#SIMULATION VARIABLES
time = 10e-9               #Simulation Runtime in Seconds
steps = 1000               #Number of Steps Taken in Simulation
dt = time/steps            #Simulation Step Length in Seconds

#PREPARING THE SIMULATION
exiting = 0         #The total number of particles that have exited the gas box
f = 0               #Used to calculate Force/Second later in the simulation
l = 0               #Amount That has bounced inside the box 
s = L/4             #Length from edge to escape hold (engine)

#RUNNING THE CONDITIONAL INTEGRATION LOOP

def particle_sim(x,v,l,exiting,f):
    for i in trange(int(steps)):
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
    guess_boxes = 1.6e13
    total_force = mean_force * guess_boxes 
    rocket_mass =  mission.spacecraft_mass
    fuel_mass = N*guess_boxes*m + rocket_mass*20
    total_mass = fuel_mass+rocket_mass
    a = -total_force/total_mass
    fuel_consume = box_mass * guess_boxes #Consume per second 
    return fuel_consume, total_force

x,l,exiting,tf = particle_sim(x,v,l,exiting,f)

particles_per_second = exiting/time              #The number of particles exiting per second
mean_force = -tf                            #The box force averaged over all time steps
box_mass = particles_per_second*m                     #The total fuel loss per second

fuel_consume,total_force = rocketengine_perf(mean_force)
print('There are {:g} particles exiting the gas box per second.'\
.format(particles_per_second))
print('The gas box exerts a thrust of {:g} N.'.format(mean_force))
print('The box has lost a mass of {:g} kg/s.'.format(box_mass))
#print(fuel_consume)

def gravity(r):
    f = (G*homeplanet_mass)/(r**2)
    return f
vesc = np.sqrt((G*2*homeplanet_mass)/(8961.62))

def orbit_launch(F,M,fuel):
    dist = 8.961621E6
    m_time = 1200
    steps = 1e5 
    dt = m_time/steps 
    v=0
    fc = 0
    while v <= vesc:
        fc += fuel_consume *dt 
        M = M - fc         
        a = F/M - gravity(dist)
        v = v + a*dt
        dist += v*dt
        pos.append(dist)
        time1.append(dt)
        print(v)
    return vel , time1, pos 

vel, time1, pos = orbit_launch(total_force,spacecraft_mass,fuel_consume)

print(time1)

print(total_force)






    

