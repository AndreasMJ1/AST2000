import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sp
import ast2000tools.constants as const
from tqdm import trange
from numba import njit
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

img = Image.open('Del4/360/23.png') # Open existing png
pixels = np.array(img) # png into numpy array
width , length = img.size

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
    error = np.zeros(360)
    for a in trange(360):
        theta , phi = HVAFAENIHELVETE(X,Y,a)
        for i in range(length):
            if i == (length-1) and k == (width-1):
                error[a] = np.sum((pixels - canvas))
            for k in range(width):
                ind = mission.get_sky_image_pixel(theta[i,k],phi[i,k])
                color = colormap[ind]
                r = color[2] ;g=  color[3] ; b= color[4]
                canvas[i,k] = np.array([r,g,b])
    ang = np.min(error)
    pos = np.where(ang == error)[0]
    return pos[0]
      

def rad_velocit():  

    def formula(lamb):
        return (const.c_AU_pr_yr*lamb)/(656.3)

    radvel_sun1 = formula(0.003300753110566649)
    radvel_sun2 = formula(0.0062612532661168565)

    radvel_plan1 = formula(0)
    radvel_plan2 = formula(0)

    ang1 = mission.star_direction_angles[0]*np.pi/180
    ang2 = mission.star_direction_angles[1]*np.pi/180
    
    kart = 1/np.sin(ang2-ang1)*np.array(([np.sin(ang2),-np.sin(ang1)],[-np.cos(ang2),np.cos(ang1)]))
    x,y = np.matmul(kart,np.array(([radvel_sun1-radvel_plan1],[radvel_sun2-radvel_plan2])))

    return x , y 

x,y = rad_velocit()

def lambda_to_velocity(lamds,phi):
    l0 = 656.3
    v = const.c*(lamds-l0)/l0
    vx = np.cos(phi)*v
    vy = np.sin(phi)*v
    return vx,vy
    
def trilateration(pos,n):
    pos_array = pos 
    phi = np.linspace(0,2*np.pi,n)
    p1 = pos_array[:,0] ; p2 = pos_array[:,1] ; p3 = pos_array[:,2]
    r = mission.measure_distances
    r1 = r[0] ; r2 = r[1] ; r3 = r[2]
    p1a = np.zeros(n) ; p2a = np.zeros(n) ; p3a = np.zeros(n)

    for i in range(n):
        p = phi[i]
        p1a[i] = (p1+np.array((r1*np.cos(p),r1*np.sin(p))))
        p2a[i] = (p2+np.array((r2*np.cos(p),r2*np.sin(p))))
        p3a[i] = (p3+np.array((r3*np.cos(p),r3*np.sin(p))))
    
    ang1 = np.isclose(p1a,p2a) ; ang2 = np.isclose(p1a,p3a)
    ang = np.where(ang1==ang2)
    pos = np.array((p1[0]+np.cos(ang)*r1,p1[1]+np.sin(ang)*r1))
    return pos 

#lambda_to_velocity(lambds,angle(phi))

#trilateration(##planetpositions,400)

print(x[0],y[0])