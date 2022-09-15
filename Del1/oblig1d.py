import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sp
import ast2000tools.constants as const
from tqdm import trange
from numba import jit
import ast2000tools.utils as utils

from ast2000tools.space_mission import SpaceMission


#CONSTANTS
k = 1.38064852e-23 #Boltzmann's Constant

#PHYSICAL VARIABLES
T = 3e3           #Temperature in Kelvin
L = 10e-6         #Box Length in meters
N = 1e3           #Number of particles
m = 3.3474472e-27 #Mass of individual particle in kg
seed = utils.get_seed('andrmj')
mission = SpaceMission(seed)

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
"""
u = np.zeros([N])    
for i in range(N):
    u[i] = np.linalg.norm(v[i])
a= np.sort(u)

def P_v(v,m,T):
    p = (m/(2*np.pi*k*T))**(3/2)*np.exp(-0.5*(m*v**2)/(k*T))*4*np.pi*v**2
    return p 
av = P_v(a,m,T)
print(l)
print(exiting)
plt.plot(a,av)
plt.show()
"""

def rocketengine_perf(mean_force,dv):
    guess_boxes = 1e18 
    
    total_force = mean_force * guess_boxes 
    rocket_mass =  mission.spacecraft_mass
    fuel_mass = N*guess_boxes*m + rocket_mass*20
    total_mass = fuel_mass+rocket_mass
    a = -total_force/total_mass
    delta_t = dv/a
    fuel_consume = box_mass * guess_boxes * delta_t 
    
    return fuel_consume, delta_t



x,l,exiting,f = particle_sim(x,v,l,exiting,f)

particles_per_second = exiting/time  #The number of particles exiting per second
mean_force = f/steps                             #The box force averaged over all time steps
box_mass = particles_per_second*m                     #The total fuel loss per second

print(rocketengine_perf(mean_force,1000))
print('There are {:g} particles exiting the gas box per second.'\
.format(particles_per_second))
print('The gas box exerts a thrust of {:g} N.'.format(mean_force))
print('The box has lost a mass of {:g} kg/s.'.format(box_mass))
print(mission.spacecraft_mass)






