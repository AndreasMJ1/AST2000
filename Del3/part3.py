from launch_func import* 
from orbit_sim import* 


#mission.set_launch_parameters(mean_force*1.6e13, fuel_consume, spacecraft_mass, 500, pos0, 0)
#mission.launch_rocket()


#x,l,exiting,tf = particle_sim(x,v,l,exiting,f)
#vel, time1, pos,spmass = orbit_launch(mean_force*1.6e13,spacecraft_mass,fuel_consume)

def gen_launch(time, posit, planet,thrust,fuelc,mass):
    esc = np.sqrt(2*G*p_masses[planet]/p_radii[planet])
    v = 0
    pos = 0 
    timer = 0

    def gravity(r,ma):
        r_norm = np.linalg.norm(r)
        f = G*p_masses[planet]*ma*1.9891e30/(r**2)
        return f 

    dt = 0.1
            
    #while v <= esc:
    for k in range(1):
        timer += dt 
        mass -= fuelc*dt  

        gravster = gravity((p_radii[planet]*1e6+pos),mass) 
        print(gravster)
        force = thrust -gravster 
        a = force / mass
        v = v + a*dt 
        pos = pos + v*dt 
        
    print(pos,v,mass,timer)

    
gen_launch(0,0,0,1684370.16453053203, 38.206316243609606,16500.0)

