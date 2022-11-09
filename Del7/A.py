###EGEN KODE###

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sp
import ast2000tools.constants as const
from tqdm import trange
from numba import jit
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from atmosphere import func 

seed = utils.get_seed('andrmj')

mission = SpaceMission(seed)
system = SolarSystem(seed)

ang_vel = 2*np.pi/utils.yr_to_s(system.rotational_periods[2])

def grav(r):
    height = system.radii[2]*1e3 + r 
    f = system.masses[2]*const.m_sun*const.G*mission.lander_mass/(height**2)
    return -f

def rotation_drag(r,A):
    height = system.radii[2] + r 
    vdrag = 0.5*func(r)*A*(ang_vel*height)**2
    return -vdrag

def radial_drag(r,v):
    vel = np.linalg.norm(v)
    drag = 0.5*func(r)*vel**2
    return -drag 

def terminal_surface():
    A = grav(0) / (radial_drag(0,3))
    return A

def thruster_force(r,v):
    thrust = grav(r) - (0.5*func(r)*118.89*(v**2-9))
    return -thrust 

print(terminal_surface())


