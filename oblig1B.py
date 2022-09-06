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
#print(v)
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
for i in trange(int(steps)):
    '''
    Each loop must begin by allowing each particle to move depending on its velocity
    '''
    x += dt*v    
    '''
    Now you need to check which particles are colliding with a wall and make them bounce:
    you may do this the slow but easy way: (1) looping over particles and using if tests:
    '''
    for j in range(int(N)):
        for u in range(3):
            if x[j][u] >= L:
                v[j][u] = -v[j][u]
                l += 1
            elif x[j][u] <=0:
                v[j][u] = -v[j][u]
                l += 1 
        
        '''
        (you may speed up this using Numba's @jit
        decorator; this may be easier for those familiar with @jit, but first-time
        users should first read the Numba chapter of our Numerical Compendium.)
        '''

    '''
    OR YOU CAN check for collisions (2) the fast and elegant way using vectorization with masking:
    Look at some examples of masking in the beginning of the Numpy section in the Numerical Compendium
    in addition to more detailed explanations inside the Numpy section.
    '''
    #collision_points=np.logical_or() # 
    #collisions_indices= np.where(collision_points == True)
    #v[collisions_indices] #do whatever you need with this one

    '''
    To check that these conditions are fulfilled for your particles, you can use
    NumPy's logical operators, which return an array of booleans giving an
    elementwise evaluation of these conditions for an entire matrix.  You may
    need:

        (a)     np.logical_or(array1, array2)            or
        (b)     np.logical_and(array1, array2)          and
        (c)     np.less(array1, array2)                   <
        (d)     np.greater(array1, array2)                >
        (e)     np.less_equal(array1, array2)            <=
        (f)     np.greater_equal(array1, array2)         >=

    '''

    '''Now, in the same way as for the collisions, you need to count the particles
    escaping, again you can use the slow way or the fast way.
    For each particle escaping, make sure to replace it!
    Then update the total force from each of the exiting particles:
    '''
        
#f+ = #fill in

u = np.zeros([N])    
for i in range(N):
    u[i] = np.linalg.norm(v[i])
a= np.sort(u)

def P_v(v,m,T):
    p = (m/(2*np.pi*k*T))**(3/2)*np.exp(-0.5*(m*v**2)/(k*T))*4*np.pi*v**2
    return p 
av = P_v(a,m,T)
print(l)
plt.plot(a,av)
plt.show()

particles_per_second = exiting/time  #The number of particles exiting per second
mean_force = f/steps                             #The box force averaged over all time steps
box_mass = particles_per_second*m                     #The total fuel loss per second

print('There are {:g} particles exiting the gas box per second.'\
.format(particles_per_second))
print('The gas box exerts a thrust of {:g} N.'.format(mean_force))
print('The box has lost a mass of {:g} kg/s.'.format(box_mass))
