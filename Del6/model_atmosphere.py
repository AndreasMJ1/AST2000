from cv2 import RHO
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sp
import ast2000tools.constants as const
from tqdm import trange
from numba import jit
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from scipy.interpolate import interp1d
seed = utils.get_seed('andrmj')

mission = SpaceMission(seed)
system = SolarSystem(seed)

k = const.k_B #Boltzman
c = const.c
u = 1.661e-27
g = const.G
plan_mass = system.masses[2] *const.m_sun
rho0 = system.atmospheric_densities[2]
h = const.m_p * 1.00784

mean_weight = 44.01069 # finn eksakt 
#44.0124 44.009

def grav(r):
    return g*plan_mass/(system.radii[2]*1e3+r)**2

def goof_func():
    ### init values ###
    r = 0
    dr = 1
    T0 = 325
    T = [T0]
    rho = [rho0]
    lf = 1.4 
    for i in range(150_000-1):
        if T[-1] >= T0/2:
            dT = -(lf-1)* mean_weight*h*grav(r)/(k*lf)
            ouf = T[-1] + dT*dr
            T.append(ouf)
            drho = -rho[-1]/T[-1] *dT -rho[-1]/T[-1]*mean_weight*h*grav(r)/(k)
            step = rho[-1] + drho * dr 
            rho.append(step)
            r += dr 
        else:
            T.append(T0/2)
            drho = -rho[-1] /(T0/2)*mean_weight*h*grav(r)/k
            step = rho[-1] + drho * dr 
            rho.append(step)
            r += dr
    pos = np.linspace(0,r,150_000)
    return T, rho , pos
T, rho , pos = goof_func()

func = interp1d(pos,rho, kind ="quadratic")


if __name__ =="__main__":

    plt.plot(pos,rho)
    plt.xlabel("Height (m)")
    plt.ylabel("Atmospheric density")
    plt.show()
    plt.plot(pos,T)
    plt.xlabel("Height (m)")
    plt.ylabel("Temprature (Kelvin)")
    plt.show()

