# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 14:27:56 2020

@author: nebue
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Dimensionless NFW density profile
#------------------------------------------------------------------------------
def rho_tilde_NFW(r_tilde):
    return r_tilde**(-1) * (1+ r_tilde)**(-2)


# Dimensionless NFW mass profile
#------------------------------------------------------------------------------
def M_tilde(r_tilde):
    return -r_tilde/(1+r_tilde) + np.log(1+r_tilde)


# Dimensionless discretized density
#------------------------------------------------------------------------------
def rho_tilde(M1,M2,r1,r2):
    return 3 * (M2-M1)/(r2**3 - r1**3)


# Dimensionless discretized pressure
#------------------------------------------------------------------------------
def p_tilde_integrand(r_tilde):
    return M_tilde(r_tilde) * rho_tilde_NFW(r_tilde) * r_tilde**(-2)


def p_tilde(r1,r2):
    lower_bound = (r1 + r2)/2
    return quad(p_tilde_integrand, lower_bound, math.inf)[0]


# Dimensionless discretized luminosity
#------------------------------------------------------------------------------
def L_tilde(r_tilde,u1,u2,p1,p2,r1,r2,sigma_hat,a = (16/np.pi)**(1/2),b = (25 * (np.pi)**(1/2) )/32,C=290/385):
    piece_1 = ( np.sqrt( (2/3) * u1 ) + np.sqrt( (2/3) * u2 ) )/2
    piece_2 = ( ( 2/(p1+p2) ) * (b/C) + (a * sigma_hat**2) )**(-1)
    piece_3 = ( 2 * (u2 - u1) )/( r2 - r1 )
    return -r_tilde**2 * piece_1 * piece_2 * piece_3


# Dimensionless discretized change in specific energy
#------------------------------------------------------------------------------
def delta_u_tilde(L1,L2,M1,M2,dt):
    return -(L2-L1)/(M2-M1) * dt

# Constants and scales
#------------------------------------------------------------------------------
G = 4.302e-06 #(kpc/M_sun)(km/s)**2
rs = 6.5 #kpc, Essig pg. 2
ps = 1.28e+7 #M_sun/kpc^3, Essig pg. 2
M0 = 4 * np.pi * rs**3 * ps
sigma_cgs = 3 #cm^2/g, Essig pg.2
convert_sigma = (1/3.086e+21)**2 * (1.989e+33/1)
sigma = sigma_cgs * convert_sigma
sigma_hat = sigma/(4 * np.pi * rs**2 * M0**(-1))
v0 = ( (G * M0)/rs )**(1/2)
a = (16/np.pi)**(1/2)
b = (25 * (np.pi)**(1/2) )/32
C = 290/385
G = 4.302e-06 #(kpc/M_sun)(km/s)**2
t0 = 1/(a * sigma * v0 * ps)


# Initialize dimensionless shell radii
#------------------------------------------------------------------------------
N = 400
r_tilde_vals = np.logspace(-1.5,3.5,N)
N = len(r_tilde_vals)
print(N,'\n')
print(r_tilde_vals,'\n')


# Initialize dimensionless masses
#------------------------------------------------------------------------------
M_tilde_vals = []
for i in range(0,N):
    M_tilde_vals.append( M_tilde(r_tilde_vals[i]) )
print(len(M_tilde_vals),'\n')
print(M_tilde_vals,'\n')


# Initialize dimensionless densities
#------------------------------------------------------------------------------
rho_tilde_vals = []
for i in range(0,N):
    if i==0:
        rho_tilde_vals.append( rho_tilde(0,M_tilde_vals[i],0,r_tilde_vals[i]) )
    else:
        rho_tilde_vals.append( rho_tilde(M_tilde_vals[i-1],M_tilde_vals[i],r_tilde_vals[i-1],r_tilde_vals[i]) )
print(len(rho_tilde_vals),'\n')
print(rho_tilde_vals,'\n')


# Initialize dimensionless pressures
#------------------------------------------------------------------------------
p_tilde_vals = []
for i in range(0,N):
    if i==N-1:
        p_tilde_vals.append( p_tilde(r_tilde_vals[i],r_tilde_vals[i]) )
    else:
        p_tilde_vals.append( p_tilde(r_tilde_vals[i],r_tilde_vals[i+1]) )
print(len(p_tilde_vals),'\n')
print(p_tilde_vals,'\n')


# Initialize dimensionless specific energies and 1D velocity dispersions
#------------------------------------------------------------------------------
u_tilde_vals = []
nu_tilde_vals = []
for i in range(0,N):
    u_tilde_vals.append( (3/2) * (p_tilde_vals[i]/rho_tilde_vals[i]) )
    nu_tilde_vals.append( np.sqrt( (2/3)*u_tilde_vals[i] ) )
print(len(u_tilde_vals),'\n')
print(u_tilde_vals,'\n')


# Initialize dimensionless luminosities
#------------------------------------------------------------------------------
L_tilde_vals = []
for i in range(0,N):
    if i==0:
        L_tilde_vals.append( L_tilde(r_tilde_vals[i],u_tilde_vals[i],u_tilde_vals[i+1],p_tilde_vals[i],p_tilde_vals[i+1],0,r_tilde_vals[i+1],sigma_hat) )
    elif i==N-1:
        L_tilde_vals.append( L_tilde(r_tilde_vals[i],u_tilde_vals[i],u_tilde_vals[i],p_tilde_vals[i],p_tilde_vals[i],r_tilde_vals[i-1],r_tilde_vals[i],sigma_hat) )
    else:
        L_tilde_vals.append( L_tilde(r_tilde_vals[i],u_tilde_vals[i],u_tilde_vals[i+1],p_tilde_vals[i],p_tilde_vals[i+1],r_tilde_vals[i-1],r_tilde_vals[i+1],sigma_hat) )
print(len(L_tilde_vals),'\n')
print(L_tilde_vals,'\n')


# Time evolution step, change in dimensionless specific energies
#------------------------------------------------------------------------------
delta_u_tilde_vals = []


# Plotting
#------------------------------------------------------------------------------
plt.clf()
plt.loglog(r_tilde_vals,M_tilde_vals,'k-',label='Mass')
plt.loglog(r_tilde_vals,[rho_tilde_NFW(x) for x in r_tilde_vals],'b-',label='CDM Density')
plt.loglog(r_tilde_vals,nu_tilde_vals,'g-',label='Velocity')
#plt.loglog(r_tilde_vals,rho_tilde_vals,'b--',label='SIDM Density')
plt.loglog(r_tilde_vals,np.abs(L_tilde_vals),'r-',label='SIDM Luminosity (absolute value)')
plt.ylim(1e-15,10**(1.75))
plt.xlim(1e-2,1e+3)
plt.xlabel('Dimensionless Radius')
plt.legend()
plt.show()