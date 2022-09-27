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



### 1.1 
print(pixels.shape)   

### 1.2 

theta = 70 ; phi = 70 
xmax = (2*np.sin(phi/2))/(1+np.cos(phi/2)) ; xmin = -xmax 
ymax = (2*np.sin(theta/2))/(1+np.cos(theta/2)) ; ymin = -ymax 

###

def sky_imag():
    canvas = np.zeros((480,640))
    width , length = canvas.shape
    print(width,length)

    for i in range(1):
        pass

sky_imag()

#mission.get_sky_image_pixel()