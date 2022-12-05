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

seed = utils.get_seed('andrmj')
dist = np.linalg.norm(np.array([539050.93487649 ,5120244.80184466]))

print(system.initial_velocities)