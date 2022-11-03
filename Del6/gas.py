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

k = const.k_B #Boltzman

def max_shift(f0,molar_mass):
    dv = 10_000 #m/s
    particle_v = np.sqrt(k*temp)/molar_mass #Bruker vi 3???
    maxshift = ((dv+particle_v) * f0)/c  #f0 = emitted frequency
    return maxshift 

def gaussian_line(fmin,sigma,lambda0,lamda):
    F = 1+ (fmin-1) *np.exp(-0.5*((lamda-lambda0)/(sigma)))
    return F 

def sigmasolver(lambda0, m, T):
    sig = (lambda0/(c*np.sqrt((k*T)/m)))
    return sig 

def GaussianModel(lambda0,mass):
    Temps = np.linspace(150,450,30)
    f_min = np.linspace(0.7,1,30)
    particle_doppler = max_shift(lambda0,mass)
    lambda_range = np.linspace(lambda0-particle_doppler,lambda0+particle_doppler,60)
    ind0 , ind1 = 1,2# FINN upper lower index 
    lowest = 3000
    for i in range(len(lambda_range)):
        for j in range(len(Temps)):
            sigma = sigmasolver(lambda_range[i],mass,Temps[j]) #HUH?
            for k in range(len(f_min)):
                chi = np.sum(((lambdas[ind0,ind1]-gaussian_line(f_min[k],sigma,lambda_range[i],lambdas[ind0,ind1]))/sigma[ind0,ind1])**2)
                if chi < lowest:
                    lowest = chi
                    values = np.array((lamda_range[i],Temps[j],f_min[k]))
    return loest , values

if __name___ == '__main__':
    o2 = [632,690,760] ; h20 = [720,820,940]
    co2 = [1400,1600]  ; ch4 = [1660,2200]
    co = [2340]        ; n2o = [2870]
    comps = [o2 , h20, co2, ch4, co, n2o]

    for mols in comps:
        for lamda_0ref in mols:


    """
    Tester 02 for 632 , 690 , 760 osv...

    """