# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 08:54:41 2020

@author: nebue
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Function to get p_NFW
#------------------------------------------------------------------------------
def p_NFW (r,rs,ps):
    Xs = r/rs
    return ps * (Xs)**(-1) * (1 + Xs)**(-2)

# Function to get M_NFW
#------------------------------------------------------------------------------
def M_NFW(r,rs,ps):
    Xs = r/rs
    if Xs>1e-5:
        return 4*np.pi*ps*rs**3*(np.log(1 + Xs)-(Xs)/(1 + Xs))
    else:
        return 4*np.pi*ps*rs**3*((1/2)*Xs**2 - (2/3)*Xs**3 + (3/4)*Xs**4 - (4/5)*Xs**5)
    
def P_integrand(r,rs,ps):
    return M_NFW(r,rs,ps) * p_NFW (r,rs,ps) * r**(-2)

def P_NFW(r,rs,ps,G = 4.302e-06):
    integral = quad(P_integrand, r, math.inf, args=(rs,ps))[0]
    return G * integral

def L_tilde_fun(r,v1,v2,p1,p2,r_upper,r_lower,a=(16/np.pi)**(1/2),b = (25 * (np.pi)**(1/2) )/32,C = 290/385):
    nu = (v2+v1)/2
    rho = (p2+p1)/2
    bracket = ( (a/b)*sig_tilde**2 + (1/C) * (1/rho) * (1/nu)**2 )**(-1)
    deriv = (v2**2 - v1**2)/((r_upper-r_lower)/2)
    return (-3/2) * r**2 * nu * bracket * deriv

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
sig_tilde = sig_m/(4 * np.pi * rs**2 * M0**(-1))
v0 = ( (G * M0)/rs )**(1/2)
t0 = 1/(a * sig_m * v0 * ps)

# Initialize shell radii values
#------------------------------------------------------------------------------
N = 400
r_tilde = np.logspace(-2,2,N)

# Initialize density, mass, pressure, luminosity values
#------------------------------------------------------------------------------
rho_tilde = [0]
for i in range(1,N):
    r_mid = ( r_tilde[i] + r_tilde[i-1] )/2
    rho = p_NFW (r_mid,rs,ps)
    rho_tilde.append(rho/ps)

M_tilde = []
for i in range(N):
    M = M_NFW(r_tilde[i],rs,ps)
    M_tilde.append(M/M0)

nu_tilde = [0]  
u_tilde = [0]
for i in range(1,N):
    r_mid = ( r_tilde[i] + r_tilde[i-1] )/2
    nu = np.sqrt( P_NFW(r_mid,rs,ps)/p_NFW (r_mid,rs,ps) )
    u = (3/2) * nu**2
    nu_tilde.append(nu/v0)
    u_tilde.append(u/v0**2)

L_tilde=[0]
for i in range(1,N-1):
    L = L_tilde_fun(r_tilde[i],nu_tilde[i],nu_tilde[i+1],rho_tilde[i],rho_tilde[i+1],r_tilde[i+1],r_tilde[i-1])
    L_tilde.append(L)