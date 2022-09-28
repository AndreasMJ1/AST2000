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
width , length = img.size



### 1.1 
#print(pixels.shape)   

### 1.2 

theta = (70*np.pi)/180 ; phi = (70*np.pi)/180 
xmax = (2*np.sin(phi/2))/(1+np.cos(phi/2)) ; xmin = -xmax 
ymax = (2*np.sin(theta/2))/(1+np.cos(theta/2)) ; ymin = -ymax 

x = np.linspace(xmin,xmax,640)
y = np.linspace(ymax,ymin,480)
X,Y = np.meshgrid(x,y)

def HVAFAENIHELVETE(X,Y,phi0):
    theta0 =np.pi/2 ; phi0 = phi0*np.pi/180 
    ro = np.sqrt(X**2+Y**2)
    beta = 2*np.arctan(ro/2)
    theta = theta0-np.arcsin((np.cos(beta)*np.cos(theta0))+((Y/ro)*np.sin(beta)*np.sin(theta0)))
    phi = phi0 + np.arctan((X*np.sin(beta))/((ro*np.sin(theta0)*np.cos(beta))-(Y*np.cos(theta0)*np.sin(beta))))
    return theta , phi 

def sky_imag():
    
    colormap = np.load('Del4\himmelkule.npy')
    pix = np.array(colormap)
    canvas = np.zeros((length,width,3),dtype ='uint8')
    for a in range(360):
        theta , phi = HVAFAENIHELVETE(X,Y,a)
        for i in range(length):
            if i == (length-1) and k == (width-1):
                img2 = Image.fromarray(canvas)
                img2.save(f'C:/Users/andym/Documents/GitHub/AST2000/Del4/360/{a}.png')
            for k in range(width):
                ind = mission.get_sky_image_pixel(theta[i,k],phi[i,k])
                color = colormap[ind]
                r = color[2] ;g=  color[3] ; b= color[4]
                canvas[i,k] = np.array([r,g,b])
 
    

sky_imag()                 