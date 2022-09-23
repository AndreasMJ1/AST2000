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

seed = utils.get_seed('andrmj')
mission = SpaceMission(seed)
system = SolarSystem(seed)

ecc = system.eccentricities
aph_ang = system.aphelion_angles
m_ax = system.semi_major_axes
p_pos = system.initial_positions



def analytic_orbits(a,e,init_ang,N,x,y):
    b = np.sqrt((1-e**2)*a**2)
    theta = np.linspace(0,np.pi*2,N)
    r = (a*(1-e**2)/(a+(e*np.cos(theta-init_ang))))
    coor = np.array([r,r]) 
    coor_k = [a*coor[0]*np.cos(theta+init_ang),b*coor[1]*np.sin(theta+init_ang)]
    #coor_k = [a*e + coor[0]*np.cos(theta+init_ang),coor[1]*np.sin(theta+init_ang)]
    return coor_k


"""
    
for i in range(7):
    plt.plot(analytic_orbits(m_ax[i],ecc[i],aph_ang[i],5000,p_pos[0][i],p_pos[1][i])[0],analytic_orbits(m_ax[i],ecc[i],aph_ang[i],5000,p_pos[0][i],p_pos[1][i])[1])
plt.scatter(0,0,color = 'black')
plt.show()
"""