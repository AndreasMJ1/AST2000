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
p_vel = system.initial_velocities

radius = 8961621.4961
def flux_rec(dist):
        total_flux = 4*np.pi*((system.star_radius*1e3)**2)*const.sigma*system.star_temperature**4
        #print(total_flux)
        fluxg = total_flux/(4*np.pi*dist**2)
        return fluxg
        
def temps():
        temps = np.zeros(7)
        for i in range(7):
            temps[i] = np.power((flux_rec(utils.AU_to_m(np.linalg.norm([p_pos[0,i],p_pos[1,i]])))/(const.sigma)),1/4)
        return temps 


if __name__ == '__main__': 
    ### 1.1 
    #flux_rec()
    print(flux_rec(utils.AU_to_m(p_pos[0,0])))
    fr = flux_rec(utils.AU_to_m(p_pos[0,0]))

    ### 1.2 
    sqr_m_solarpanel = (40/0.12)/2900 
    print(sqr_m_solarpanel)      

    ### 1.3 
    #flux_rec(r)*2*np.pi*R**2

    ### 1.4
    #temps()

    ### 1.5 
    ts = temps()
    print(ts)     #habitable = #3 , #7
    flux_rec(utils.AU_to_m(p_pos[0,0]))