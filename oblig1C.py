import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sp
import ast2000tools.constants as const
from tqdm import trange
from numba import njit

#CONSTANTS
k = 1.38064852e-23          #Boltzmann's Constant

#PHYSICAL VARIABLES
T = 3e3          #Temperature in Kelvin
L = 10e-6              #Box Length in meters
N = 100            #Number of particles
m = 3.3474472e-27           #Mass of individual particle in kg

#INITIAL CONDITIONS
sigma =  np.sqrt((k*T)/m)    #The standard deviation of our particle velocities
 
x =  np.random.uniform(0,L, size = (int(N), 3))                     # Position vector, fill in uniform distribution in all 3 dimensions
v =  np.random.normal(0,sigma, size = (int(N), 3))                       # Velocity vector, fill in correct distribution

'''

An array with 10 particles (such that N = 10) would look like this:

                      x =  [[x0, y0, z0],
                            [x1, y1, z1],
                            [x2, y2, z2],
                            [x3, y3, z3],
                            [x4, y4, z4],
                            [x5, y5, z5],
                            [x6, y6, z6],
                            [x7, y7, z7],
                            [x8, y8, z8],
                            [x9, y9, z9]]
'''

#SIMULATION VARIABLES
time = 10e-9                       #Simulation Runtime in Seconds
steps = 1000               #Number of Steps Taken in Simulation
dt = time/steps                  #Simulation Step Length in Seconds

#PREPARING THE SIMULATION
exiting = 0         #The total number of particles that have exited the gas box
f = 0                   #Used to calculate Force/Second later in the simulation



#RUNNING THE CONDITIONAL INTEGRATION LOOP
l = 0
s = L/4
for i in trange(int(steps)):
    x += dt*v    
     
    for j in range(int(N)):
        if s <= x[j][0] <= 3*s and s <= x[j][1] <= 3*s and x[j][2] <= 0:
            exiting+=1
            f += v[j][2]*m/dt
            x[j] =  np.random.uniform(0,L, size = (1, 3))                     # Position vector, fill in uniform distribution in all 3 dimensions
            v[j] =  np.random.normal(0,sigma, size = (1, 3)) 
        else:
            for u in range(3):
                if x[j][u] >= L or x[j][u] <=0:
                    v[j][u] = -v[j][u]
                    l += 1
            
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

particles_per_second = exiting/time  #The number of particles exiting per second
mean_force = f/steps                             #The box force averaged over all time steps
box_mass = particles_per_second*m                     #The total fuel loss per second

print('There are {:g} particles exiting the gas box per second.'\
.format(particles_per_second))
print('The gas box exerts a thrust of {:g} N.'.format(mean_force))
print('The box has lost a mass of {:g} kg/s.'.format(box_mass))
