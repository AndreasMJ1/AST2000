import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sp
import ast2000tools.constants as const
from tqdm import trange
from numba import njit
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission

seed = utils.get_seed('andrmj')
mission = SpaceMission(seed)
system = SolarSystem(seed)

plan_pos = np.load("planet_trajectories.npz")
times = plan_pos["times"]
a = 1494

time = 1494 * times[1]
print(time)

print(system.star_radius)
print(system.star_mass)
print(system.star_temperature)
print(system.star_color)

seed = utils.get_seed('andrmj')
print(seed)