from launch_func import* 

mission.set_launch_parameters(mean_force*1.6e13, fuel_consume, spacecraft_mass, 500, pos0, 0)
mission.launch_rocket()

vel, time1, pos = orbit_launch(mean_force*1.6e13,spacecraft_mass,fuel_consume)

print(time1[-1])
print(vel)
