#IKKE KODEMAL

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

seed = 73494

mission = SpaceMission(seed)
system = SolarSystem(seed)

ang_vel = 2*np.pi/utils.yr_to_s(system.rotational_periods[2])

### Creating functions for preparation before landing 

def grav(r):
    """
    Gravitational force applied onto the lander at a given r (m)
    """
    height = system.radii[2]*1e3 + r 
    f = system.masses[2]*const.m_sun*const.G*mission.lander_mass/(height**2)
    return -f

def rotation_drag(r,A):
    """
    The sideways drag produced upon the lander at a given distance, r. Drag produced by the movement of the 
    atmosphere 
    """
    height = system.radii[2] + r 
    vdrag = 0.5*func(r)*A*(ang_vel*height)**2
    return -vdrag

def radial_drag(r,v):
    """
    The radial drag experienced by the lander at a given distance, r.
    """
    vel = np.linalg.norm(v)
    drag = 0.5*func(r)*vel**2
    return -drag 

def Parachute_area():
    """
    The area of the parachute necessary to maintain a low stable downward velocity. 
    """
    A = grav(0) / (radial_drag(0,3))
    return A

def thruster_force(r,v):
    """
    The thrust force needed for the lander to slow down to a velocity of 3 m/s
    """
    thrust = grav(r) - (0.5*func(r)*118.89*(v**2-9))
    return -thrust 

print(Parachute_area())


