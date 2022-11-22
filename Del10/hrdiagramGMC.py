import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.constants as const
from ast2000tools.star_population import StarPopulation
import ast2000tools.utils as utils
seed = utils.get_seed('andrmj')
from ast2000tools.solar_system import SolarSystem
system = SolarSystem(seed)

stars = StarPopulation()
T = stars.temperatures # [K]
L = stars.luminosities # [L_sun]
r = stars.radii        # [R_sun]

c = stars.colors
s = np.maximum(1e3*(r - r.min())/(r.max() - r.min()), 1.0) # Make point areas proportional to star radii

fig, ax = plt.subplots()
ax.scatter(T, L, c=c, s=s, alpha=0.8, edgecolor='k', linewidth=0.05)

ax.set_xlabel('Temperature [K]')
ax.invert_xaxis()
ax.set_xscale('log')
ax.set_xticks([35000, 3000, 10])
ax.set_xticklabels(list(map(str, ax.get_xticks())))
ax.set_xlim(40000, 5)
ax.minorticks_off()

ax.set_ylabel(r'Luminosity [$L_\odot$]')
ax.set_yscale('log')
ax.set_ylim(1e-4, 1e6)
ax.set_title("Hertzsprung-Russell-diagram")

T_star = 6784 #Kelvin
L_star = 3.29 #luminositet 

ax.scatter(T_star, L_star, s = 100, c = "k", marker = ".", linewidth = 0.01)

T_gmc = 10 #K
L_gmc = 39.14 #luminositet

ax.scatter(T_gmc, L_gmc, s = 200, c = "k", marker = ".", linewidth = 0.01)

plt.show()
#plt.savefig('HR_diagram.png')

