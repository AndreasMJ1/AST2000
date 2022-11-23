#IKKE KODEMAL

#Part 1

#A1
#Function for the normal probability distribution

#1.1
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sp
import ast2000tools.constants as const
#Inital values 
mu = 0 
sigma = 1 

def f(x):
    return (1 /(np.sqrt(2*np.pi)*sigma)) * np.exp(-0.5 *((x-mu)/sigma)**2)

def P(f,a,b):
    P = sp.quad(f, a, b)[0]
    return P

#1.3
sigmal = [sigma*1,sigma*2,sigma*3] #Definerer ulike standardavvik

for i in sigmal:
    print(P(f,-i,i))

#0.682689492137086
#0.9544997361036417
#0.9973002039367399


#2.1
#Inital values 
N = 10**5
T = 3000 
k = const.k_B
m = const.m_H2


sigma = np.sqrt((k*T) /m)

vx = np.linspace(-2.5*10**(4), 2.5*10**(4),N)

plt.plot(vx, f(vx))
plt.title('Sannsynlighetstetthet til hastighet i x-retning (10^4 m/s)')
plt.xlabel('Hastighet')
plt.ylabel('Sannsynlighet')
plt.show()

#2.2
print(P(f,5*10**(3), 30*10**(3)))

#2.3
vx = np.linspace(0, 3*10**(4),N)

def P_v(v,m,T):
    p = (m/(2*np.pi*k*T))**(3/2)*np.exp(-0.5*(m*v**2)/(k*T))*4*np.pi*v**2
    return p 

v_x = P_v(vx,m,T)


plt.plot(vx, (v_x))
plt.title('Sannsynlighetstetthet til hastighet i x-retning (10^4 m/s)')
plt.xlabel('Hastighet')
plt.ylabel('Sannsynlighet')
plt.show()