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
from ast2000tools.relativity import RelativityExperiments
import scipy.constants as sp 


seed = utils.get_seed('andrmj')

mission = SpaceMission(seed)
system = SolarSystem(seed)
relativity = RelativityExperiments(seed)

#relativity.spaceship_duel(2)

#relativity.spaceship_race(2)

#relativity.antimatter_spaceship(2)

m = 1e6

c = const.c

v = 0.148647 *c

h = sp.h 

N = 5.66418e41
gamma = 1/(np.sqrt(1-(v**2/c**2)))
e = 2*(gamma*m*c**2) # +0.5*m*v**2
u = e/N
print((h*c/u)*1e9)
