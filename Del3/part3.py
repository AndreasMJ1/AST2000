from launch_func import* 


mission.set_launch_parameters(mean_force*1.6e13, fuel_consume, spacecraft_mass, 500, pos0, 0)
mission.launch_rocket()


x,l,exiting,tf = particle_sim(x,v,l,exiting,f)
vel, time1, pos,spmass = orbit_launch(mean_force*1.6e13,spacecraft_mass,fuel_consume)

def gen_launch(time, pos, planet):
    vesc = 
    while v >= vesc:



#print(time1[-1])
#print(vel)
