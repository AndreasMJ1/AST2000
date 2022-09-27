import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sp
import ast2000tools.constants as const
from tqdm import trange
from numba import jit
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from PIL import Image

seed = utils.get_seed('andrmj')
mission = SpaceMission(seed)
system = SolarSystem(seed)

ecc = system.eccentricities
aph_ang = system.aphelion_angles
m_ax = system.semi_major_axes
p_pos = system.initial_positions
p_vel = system.initial_velocities
p_masses = system.masses
p_radii = system.radii


img = Image.open('Del4\sample0000.png') # Open existing png
pixels = np.array(img) # png into numpy array
width = len(pixels[0, :])
#redpixs = [(255, 0, 0) for i in range(width)] # Array of red pixels
#pixels[500, :] = redpixs # Insert into line 500

print(width)
#img2 = Image.fromarray(pixels)
#img2.save(’exampleWithRedLine.png’) # Make new png

