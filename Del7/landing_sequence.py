import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sp
import ast2000tools.constants as const
from tqdm import trange
from numba import jit
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from A import thruster_force

seed = utils.get_seed('andrmj')

mission = SpaceMission(seed)
system = SolarSystem(seed)

ecc = system.eccentricities
aph_ang = system.aphelion_angles
m_ax = system.semi_major_axes
p_pos = system.initial_positions
p_vel = system.initial_velocities
spacecraft_mass = mission.spacecraft_mass*15
homeplanet_dist = 1.2289822738 #AU
homeplanet_mass = 1.5616743232192E25 #7.80837162e-06 m_sun


G = 6.6743e-11 
#G = 4*np.pi**2
Sm = system.star_mass
radius = system.radii[0]*1e3

def gravity(r):
    rnorm = np.linalg.norm(r)
    f = ((G*homeplanet_mass)*r)/(rnorm**3)
    return f


def general_launch(phi,time,fuel_consume,Mass,F):
    
    dist = system.radii[0]*1e3
    phi = phi*np.pi/180
    rp = np.array((dist*np.cos(phi),dist*np.sin(phi)))
    
    rpi = np.array((dist*np.cos(phi),dist*np.sin(phi)))
    dt = 0.001
    v = np.array((0,0))
    fc = 0
    timer = 0
    F = np.array((np.cos(phi)*F,np.sin(phi)*F))
    vesc = np.sqrt((G*2*homeplanet_mass)/(np.linalg.norm(rp)))
    while np.linalg.norm(v) <= vesc:
        fc += fuel_consume*dt  
        M = Mass - fc         
        a1 = F/M
        a2 = -gravity(rp)
        a = a1 + a2
        v = v + a*dt
        rp = rp + v*dt
        timer += dt 

    explanpos = np.load("planet_trajectories.npz")
    npos = explanpos['planet_positions']
    plan_pos = np.einsum("ijk->kji",npos)
    
    velo = np.load('velocities.npy') 
    
    ind = int(time/0.0002)

    launch_point = plan_pos[ind,0,:] + utils.m_to_AU(rpi) #launch_point = plan_pos[:,0,ind] + utils.m_to_AU(rpi) 
    
    ycomp = np.cos(phi)*dist*2*np.pi/(utils.day_to_s(system.rotational_periods[0]))
    ypos = np.array((0,utils.m_to_AU(ycomp*timer)))
    pos_p = np.array((velo[ind,0,0]*utils.s_to_yr(timer),velo[ind,0,1]*utils.s_to_yr(timer)))
    fin_pos = plan_pos[ind,0,:] + utils.m_to_AU(rp) + ypos + pos_p

    inc = np.array((-9.49991765 ,-0.01766334)) - np.array([-9.28709058, -0.13534456])
    
    print(inc, "boost")
    return rp ,fin_pos, timer ,launch_point


if __name__ == '__main__': #1494*0.0002
    explanpos = np.load("planet_trajectories.npz")
    npos = explanpos['planet_positions']
    plan_pos = np.einsum("ijk->kji",npos)
    times = explanpos['times'] 
    mean_force = 776938.7689392876
    rp, finpos, time ,launch_point = general_launch(180, times[1494] , 0.18871032743168 ,spacecraft_mass ,mean_force)
    #print(rp,finpos,time,launch_point)

    exac_pos = [8.69554508e-04 ,1.22575863e+00]
    exac_pos = [-4.39913930e-04  ,1.22575507e+00]
    R0 = np.array((plan_pos[1494,0,0],plan_pos[1494,0,1]))  # np.array((9.69192737e-04 ,1.22576046e+00))  # [9.69192737e-04 1.22576046e+00]             # AU
    R0 = R0 - np.array([radius, 0]) / const.AU  
    mission.set_launch_parameters(mean_force, 0.18871032743168, spacecraft_mass, 500, R0 , times[1494])
    mission.launch_rocket(0.001)
    mission.verify_launch_result(exac_pos) #[1.22905961e+00 8.38221473e-05]
    mission.take_picture()
    

    mission.verify_manual_orientation(exac_pos, [-9.28636344 ,-0.15831185], 192) # 192 


    ins = mission.begin_interplanetary_travel()
    

    ind = int(2000)#int(6717+266*4)#+ 1494
    boost = np.array((-9.49991765, -0.01766334)) - np.array((-9.28636344 ,-0.15831185))
    ins.boost(boost)
    ins.coast_until_time(6717*times[1])
    ins.orient()

    ins.coast_until_time(6717*times[1]+5*231*times[1])
    timep = 6717*times[1]+5*231*times[1] 
    ins.orient()
    ins.boost(np.array((1.03,0.73)))
    ins.coast_until_time(timep + 400*times[1])
    indt = int(6717+5*231+400)
    ins.orient()
    boostnr2 =  np.array((4.27,  1.3)) - np.array((5.02218,1.53139))
    ins.boost(boostnr2)

    planet = np.array((plan_pos[indt+15,2,0],plan_pos[indt+15,2,1]))
    t,v,p = ins.orient()
    ang = planet-p
    ins.boost(np.array((0.035,0.025))*np.array([ang[1],ang[0]]))
    ins.look_in_direction_of_planet(2)
    
    ins.coast(0.022)
    ins.record_destination(2)
    
    
    landing = mission.begin_landing_sequence()
    landing.adjust_parachute_area(118.89)
    
    landing.look_in_direction_of_planet(2)
    
    landing.orient()
    landing.boost(-np.array((-1162.68, 42.6937, 0))) # -1162.68, 42.6937, 0
    ang = np.arctan(1.17361e+08/1.3586e+07)
    print(ang)
    landing.boost(30*np.array((-np.cos(ang),np.sin(ang),0)))
    landing.fall(80_000)
    def adjust():

        t,p,v = landing.orient()
        landing.boost(np.array((-0.5*v[0],-0.5*v[1],0)))
        landing.fall(20_000)
        t,p,v = landing.orient()
        landing.boost(np.array((-0.5*v[0],-0.5*v[1],0)))
        landing.fall(20_000)
        t,p,v = landing.orient()
        landing.start_video()
        landing.boost(np.array((-0.8*v[0],-0.8*v[1],0)))
        landing.fall(15_000)
        t,p,v = landing.orient()
        landing.boost(np.array((-0.7*v[0],-0.7*v[1],0)))
        landing.fall(3_000)
        t,p,v = landing.orient()
        landing.boost(np.array((-0.8*v[0],-0.8*v[1],0)))
    adjust()
    landing.fall(800)
    t,p,v = landing.orient()
    
    
    t,p,v = landing.orient()
    landing.boost(np.array((-0.95*v[0],-0.95*v[1],0)))
    landing.fall(130)
    t,p,v = landing.orient()
    landing.boost(np.array((-0.95*v[0],-0.95*v[1],0)))
    landing.fall(130)
    t,p,v = landing.orient()
    landing.boost(np.array((-0.95*v[0],-0.95*v[1],0)))
    landing.fall(300)
    t,p,v = landing.orient()
    landing.boost(np.array((-v[0],-v[1],0)))
    landing.launch_lander(np.array((0,0,0)))
    landing.deploy_parachute()
    landing.fall(50)
    t,p,v = landing.orient()
    #landing.boost(np.array((-v[0],-v[1],0)))
    landing.fall(50)
    landing.orient()
    landing.fall(50_000)
    """
    landing.boost(np.array((-v[0],-v[1],0)))
    landing.deploy_parachute()
    landing.fall(50)
    t,p,v = landing.orient()
    landing.boost(np.array((-v[0],-v[1],0)))
    landing.fall(50)
    landing.orient()
    """
    

    
    landing.orient()
    landing.adjust_landing_thruster(thruster_force(500,np.linalg.norm(np.array((v[0],v[1])))),500)
    landing.finish_video()


            
