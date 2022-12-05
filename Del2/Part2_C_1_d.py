from ast import NotEq
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



seed = 73494
mission = SpaceMission(seed)
system = SolarSystem(seed)

ecc = system.eccentricities
aph_ang = system.aphelion_angles
m_ax = system.semi_major_axes
p_pos = system.initial_positions
p_vel = system.initial_velocities

p3 = system.masses[0]
G = 4*np.pi**2
Sm = system.star_mass


def gravS(r:np.asarray): 
    r_norm = np.linalg.norm(r)         #Function for gravity, vectorized
    a = -G*Sm*p3*(r/(r_norm**3))
    return a 

def sim_orbit1(steps,dt):            #Simulation Loop
    """
    Simulating orbits to calculate mechanical energy
    """
    sv = -(p3*p_vel[1,0])/Sm
    r = np.zeros((steps,2,2))
    v = np.zeros((steps,2,2))
    r0 = np.array([[0,0],[p_pos[0,0],p_pos[1,0]]])
    v0 = np.array([[0,-sv],[p_vel[0,0],p_vel[1,0]]])
    red_mass = (Sm*p3)/(Sm+p3)
    v[0] = v0
    r[0] = r0
    dt = dt
    kin=np.zeros(steps)
    pot = np.zeros(steps)
    tot = np.zeros(steps)
    kin[0] = 0.5*red_mass*np.linalg.norm(v[0,1]+v[0,0])**2
    pot[0] = G*(Sm+p3)*red_mass/np.linalg.norm(r[0,0]+r[0,1])
    tot[0] = kin[0] - pot[0] 
    for i in trange(steps-1):
        a_s = gravS(r[i,1]-r[i,0])/Sm

        a_p = gravS(r[i,1]-r[i,0])/p3

        vhs = v[i,0] + a_s*dt/2
        vhp = v[i,1] + a_p*dt/2
        
        r[i+1,0] = r[i,0] + vhs*dt
        r[i+1,1] = r[i,1] + vhp*dt 

        f_s = gravS(r[i+1,1]-r[i+1,0])
        a_s = f_s/Sm

        f_p = gravS(r[i+1,1]-r[i+1,0])
        a_p = f_p/p3

        v[i+1,0] = vhs + a_s*dt/2
        v[i+1,1] = vhp + a_p*dt/2

        kin[i+1] = 0.5*red_mass*np.linalg.norm(v[i,1]+v[i,0])**2
        pot[i+1] = G*(Sm+p3)*red_mass/np.linalg.norm(r[i,0]+r[i,1])
        tot[i+1] = kin[i] - pot[i] 
        
    return r , v , kin , pot , tot 

r1,v1 ,kin,pot,tot = sim_orbit1(119000,0.0002)
t = np.linspace(0,119000*0.0002,119000)

plt.plot(r1[:,0,0],r1[:,0,1])
plt.plot(r1[:,1,0],r1[:,1,1])
plt.show()

print(f"Largest difference ={100*abs(((np.max(tot))-(np.min(tot)))/((np.mean(tot))))}%")

plt.plot(t,kin)
plt.plot(t,pot)
plt.plot(t,tot)

plt.show()


