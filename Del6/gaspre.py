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
c = const.c
u = 1.661e-27

flux = np.load("flux_values.npy")
lambdas = np.load("lambda_values.npy")
noise = np.load("noise.npy")


def max_shift(f0,molar_mass):
    dv = 10_000 #m/s
    particle_v = np.sqrt(k*450/molar_mass) #Bruker vi 3???
    maxshift = ((dv+particle_v) * f0)/c  #f0 = emitted frequency
    return maxshift 

def gaussian_line(fmin,sigma,lamda,lambda0):
    F = 1+ ((fmin-1) *np.exp(-0.5*((lamda-lambda0)/(sigma))**2))
    return F 

def sigmasolver(lambda0, m, T):
    sig = (lambda0/c*np.sqrt((k*T/m)))
    return sig 

def GaussianModel(lambda0,mass):
    Temps = np.linspace(150,450,60)
    f_min = np.linspace(0.7,1,60)
    
    particle_doppler = max_shift(lambda0,mass)
    #lambda_range = np.linspace(lambda0-particle_doppler,lambda0+particle_doppler,60)
    
    
    tol = 1e-3
    ### FINN LAV/HØY INDEX ##
    ind0 = np.where(abs(lambdas-np.round(lambda0-particle_doppler,3))< tol)[0][0]
    ind1 = np.where(abs(lambdas-np.round(lambda0+particle_doppler,4))< tol)[0][-1]
    lambda_range = np.linspace(lambdas[ind0],lambdas[ind1],60)
    lambda_range*= 1e9
    lambda0 = lambda0*1e9
    

         ### AUGHHH ####
    lowest = 300000
    for i in range(len(lambda_range)):
        for j in range(len(Temps)):
            sigma = sigmasolver(lambdas[ind0:ind1]*1e9,mass,Temps[j]) #HUH?
            for k in range(len(f_min)):
                chi = np.sum(((flux[ind0:ind1]-gaussian_line(f_min[k],sigma,lambda_range[i],lambdas[ind0:ind1]*1e9))/noise[ind0:ind1,1])**2)
                if chi < lowest:
                    lowest = chi
                    computed = gaussian_line(f_min[k],sigma,lambda_range[i],lambdas[ind0:ind1]*1e9)
                    vals = np.array((f_min[k],lambda_range[i]*1e-9,Temps[j]))
    #plt.plot(lambdas[ind0:ind1],flux[ind0:ind1])
    #plt.plot(lambdas[ind0:ind1],computed)
    #plt.xlabel("Lambda value +- max doppler shift")
    #plt.ylabel("Relative flux")
    #plt.show()
    print(f"{vals[0]:.4f} -||- {vals[1]:.4f} -||- {vals[2]:.4f} -||- {particle_doppler:.4f}")

    return lowest

if __name__ == '__main__':

    o2 = [[632,690,760],[31.998*u]] ; h20 = [[720,820,940],[18.01528*u]]
    co2 = [[1400,1600],[44.009*u]]  ; ch4 = [[1660,2200],[16*u]]
    co = [[2340],[28.01*u]]         ; n2o = [[2870],[44.0124*u]]
    comps = [o2 , h20, co2, ch4, co, n2o]
    O2 = [[632,690,760],[31.998*u]] ; H2O = [[720,820,940],[18.01528*u]]
    CO2 = [[1400,1600],[44.009*u]]  ; CH4 = [[1660,2200],[16*u]]
    CO = [[2340],[28.01*u]]         ; N2O = [[2870],[44.0124*u]]
    comps = [O2 , H2O, CO2, CH4, CO, N2O]

    for mols in comps:
        print("--------------------------------------------------")
        for lamda_0ref in mols[0]:
            results = GaussianModel(lamda_0ref,mols[1][0])