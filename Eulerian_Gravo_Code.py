# -*- coding: utf-8 -*-
"""
Created on Thu May 28 08:27:22 2020

@author: nebue
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import quad


# Function to get p_s
#------------------------------------------------------------------------------
def get_ps(C,p_crit):
    return (200*p_crit/3) * (C**3) * (np.log(1 + C) - C/(1 + C))**(-1)


# Function to get r_s
#------------------------------------------------------------------------------
def get_rs(M_200,C,rho_s):
    return (M_200 * (4*np.pi*rho_s)**(-1) * (np.log(1 + C) - C/(1 + C))**(-1))**(1/3)


# Function to get rho_NFW
#------------------------------------------------------------------------------
def rho_NFW (r,r_s,rho_s):
    return rho_s * (r/r_s)**(-1) * (1 + r/r_s)**(-2)


# Function to get M_NFW
#------------------------------------------------------------------------------
def M_NFW(r,r_s,rho_s):
    X_s = r/r_s
    if X_s>1e-5:
        return 4*np.pi*rho_s*r_s**3*(np.log(1 + X_s) - (X_s)/(1 + X_s))
    else:
        return 4*np.pi*rho_s*r_s**3*((1/2)*X_s**2 - (2/3)*X_s**3 + (3/4)*X_s**4 - (4/5)*X_s**5)


# Function to get integrand for pressure integral
#------------------------------------------------------------------------------  
def p_integrand(r,r_s,rho_s):
    return M_NFW(r,r_s,rho_s) * rho_NFW (r,r_s,rho_s) * r**(-2)


# Function to get P_NFW
#------------------------------------------------------------------------------
def p_NFW(r,r_s,rho_s,G = 4.302e-06):
    integral = quad(p_integrand, r, math.inf, args=(r_s,rho_s))[0]
    return G * integral


# Function to calculate finite differences
#------------------------------------------------------------------------------
def derivative(y1,y2,Delta_r):
    return (y2-y1)/Delta_r


# Function to get L_NFW
#------------------------------------------------------------------------------
def L_new(r1,nu_1,nu_2,p,sigma,Delta_r,G = 4.302e-06,a = (16/np.pi)**(1/2),b = (25 * (np.pi)**(1/2) )/32,C = 290/385):
    term_1 = -12 * np.pi * r1**2 * nu_1**2 * a * b * sigma
    term_2 = ( a*sigma**2 + (b/C) * (4*np.pi*G)/(p) )**(-1)
    term_3 = derivative(nu_1,nu_2,Delta_r)
    return term_1 * term_2 * term_3


# Function to time-evolve densities
#------------------------------------------------------------------------------
def Delta_rho(Delta_t,rho_1,rho_2,u1,u2,Delta_r,r):
    bracket = -rho_1*derivative(u1,u2,Delta_r) - u1*derivative(rho_1,rho_2,Delta_r) - (2/r)*rho_1*u1
    return Delta_t*bracket


# Function to time-evolve fluid velocities
#------------------------------------------------------------------------------
def Delta_u(Delta_t,u1,u2,Delta_r,M,r,rho,p1,p2,G = 4.302e-06):
    bracket = -u1*derivative(u1,u2,Delta_r) - (G*M)/r**2 - (1/rho)*derivative(p1,p2,Delta_r)
    return Delta_t*bracket


# Function to time-evolve 1D vel. disp.
#------------------------------------------------------------------------------
def Delat_nu(Delta_t,nu_1,nu_2,r,p,L1,L2,Delta_r,u1,u2):
    bracket = (-1/(4*np.pi*r**2*p))*derivative(L1,L2,Delta_r) - (derivative(u1,u2,Delta_r) + (2*u1)/r) - (3*u1)/(nu_1)*derivative(nu_1,nu_2,Delta_r) 
    return Delta_t * (nu_1/3) * bracket


# Function to time-evolve mass
#------------------------------------------------------------------------------
def M_new(index,r_vals,rho_vals,Delta_r_vals):
    if index == 0:
        return (4*np.pi)/(3) * Delta_r_vals[0]**3 * rho_vals[0] 
    else:
        SUM = (4*np.pi)/(3) * Delta_r_vals[0]**3 * rho_vals[0] 
        for i in range(1,index+1):
            SUM+= 4 * np.pi * r_vals[i]**2 * rho_vals[i] * Delta_r_vals[i]
        return SUM


# Initialize constants and choose rho_s, r_s, and sigma_m
#------------------------------------------------------------------------------
G = 4.302e-06
a = (16/np.pi)**(1/2)
b = (25 * (np.pi)**(1/2) )/32
C = 290/385


rho_s = 1.28e+7 # M_sun/kpc^3, Essig pg. 2
r_s = 6.5 # kpc, Essig pg. 2
sigma_convert = (1/3.086e+21)**2 * (1.989e+33/1) # converts cm to kpc, grams to M_sun
sigma = 3*sigma_convert # Essig pg. 2


# Discretize halo and initialize quantities
#------------------------------------------------------------------------------
N = 400 # Nishikawa pg. 5
s_vals = np.logspace(-2,3,N+1) * r_s # Essig pg. 7
r_vals = []
Delta_r_vals = []
for i in range(N):
    r_vals.append( (s_vals[i] + s_vals[i+1])/2 )
    Delta_r_vals.append(s_vals[i+1] - s_vals[i])


rho_vals = [rho_NFW (x,r_s,rho_s) for x in r_vals]
M_vals = [M_NFW(x,r_s,rho_s) for x in r_vals]
u_vals = np.zeros(400)
p_vals = [p_NFW(x,r_s,rho_s) for x in r_vals]


nu_vals=[]
for i in range(N):
    nu_vals.append( np.sqrt(p_vals[i]/rho_vals[i]) )


L_vals=[]
for i in range(N-1):
    L_vals.append( L_new(r_vals[i],nu_vals[i],nu_vals[i+1],p_vals[i],sigma,Delta_r_vals[i]) )
    

# Time evolve quantities
#------------------------------------------------------------------------------
t_r_vals = []
for i in range (N):
    t_r_vals.append( 1/(rho_vals[i] * nu_vals[i] * sigma * a) )

epsilon_t = 1e-4
Delta_t = min(t_r_vals) * epsilon_t
print(Delta_t)
    


# Plotting
#------------------------------------------------------------------------------
plt.clf()
plt.loglog(r_vals,rho_vals,label='Density')
plt.loglog(r_vals,M_vals,label='Mass')
plt.loglog(r_vals,p_vals,label='Pressure')
plt.loglog(r_vals,nu_vals,label='1D Vel. Disp.')
plt.loglog(r_vals[:N-1],np.abs(L_vals),label='Luminosity (abs val)')
plt.xlabel('r (kpc)')
plt.legend()
plt.show()