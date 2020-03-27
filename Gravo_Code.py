# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 08:54:41 2020

@author: nebue
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Function to get p_NFW
#------------------------------------------------------------------------------
def p_NFW (r,r_s,p_s):
    X_s = r/r_s
    return p_s * (X_s)**(-1) * (1 + X_s)**(-2)

# Function to get M_NFW
#------------------------------------------------------------------------------
def M_NFW(r,r_s,p_s):
    X_s = r/r_s
    if X_s>1e-5:
        return 4*np.pi*p_s*r_s**3*(np.log(1 + X_s)-(X_s)/(1 + X_s))
    else:
        return 4*np.pi*p_s*r_s**3*((1/2)*X_s**2 - (2/3)*X_s**3 + (3/4)*X_s**4 - (4/5)*X_s**5)
    
def P_integrand(r,M,rho):
    return (M * rho)/(r)**2

def P_NFW(r,M,rho):
    integral = quad(P_integrand, 0, r, args=(M,rho))[0]
    return integral

# Initialize scales
# -----------------------------------------------------------------------------
a = (16/np.pi)**(1/2)
b = (25 * (np.pi)**(1/2) )/32
C = 290/385
G = 4.302e-06 #(kpc/M_sun)(km/s)**2

rs = 6.5 #kpc, Essig pg. 2
ps = 1.28e+7 #M_sun/kpc^3, Essig pg. 2
M0 = 4 * np.pi * rs**3 * ps
sig_m = 3 #cm^2/g, Essig pg.2
v0 = ( (G * M0)/rs )**(1/2)
t0 = 1/(a * sig_m * v0 * ps)

# Initialize shell radii values
#------------------------------------------------------------------------------
N = 400
r_tilde = np.logspace(-2,2,N)

# Initialize density, mass, pressure, luminosity values
#------------------------------------------------------------------------------
rho_tilde = []
for i in range(1,N):
    r_mid = ( r_tilde[i] + r_tilde[i-1] )/2
    rho = p_NFW (r_mid,rs,ps)
    rho_tilde.append(rho/ps)

M_tilde = []
for i in range(N):
    M = M_NFW(r_tilde[i],rs,ps)
    M_tilde.append(M/M0)

P_tilde = []  
for i in range(1,N):
    r_mid = ( r_tilde[i] + r_tilde[i-1] )/2
    P = P_NFW(r_mid,M_tilde[i],rho)
    P_tilde.append(P)