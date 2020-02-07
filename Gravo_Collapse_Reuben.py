# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 21:36:25 2020

@author: nebue
"""

import copy
import math
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
plt.clf()

# Function to get p_s
#------------------------------------------------------------------------------
def get_ps(C,p_crit):
    return (200*p_crit/3) * (C**3) * (np.log(1 + C) - C/(1 + C))**(-1)

# Function to get r_s
#------------------------------------------------------------------------------
def get_rs(M_200,C,p_s):
    return (M_200 * (4*np.pi*p_s)**(-1) * (np.log(1 + C) - C/(1 + C))**(-1))**(1/3)

# Function to get p_NFW
#------------------------------------------------------------------------------
def p_NFW (r,r_s,p_s):
    return p_s * (r/r_s)**(-1) * (1 + r/r_s)**(-2)

# Function to get M_NFW
#------------------------------------------------------------------------------
def M_NFW(r,r_s,p_s):
    X_s = r/r_s
    if X_s>1e-5:
        return 4*np.pi*p_s*r_s**3*(np.log(1 + X_s)-(X_s)/(1 + X_s))
    else:
        return 4*np.pi*p_s*r_s**3*((1/2)*X_s**2 - (2/3)*X_s**3 + (3/4)*X_s**4 - (4/5)*X_s**5)

# Laura's radial velocity dispersion
#------------------------------------------------------------------------------
def vel_disp_NFW_Laura(r,r_s,p_s,G=4302./10.**9):
    x=  r/r_s
    prefactor = 2.*G*np.pi*p_s*r_s**2.
    sigmar = np.sqrt(prefactor*(1/x)*(x*(-1.+x*(-9.-7.*x+np.pi**2.*(1.+x)**2.))-x**2.*(1.+x)**2.*np.log(x)+(1.+x)*np.log(1.+x)*(1.+x*(-3.+(-5.+x)*x)+3.*x**2.*(1.+x)*np.log(1.+x))+6.*x**2.*(1.+x)**2.*float(mp.polylog(2.,-x))))
    return sigmar
#------------------------------------------------------------------------------

# Reuben's radial velocity dispersion
#------------------------------------------------------------------------------
def jeans_integrand(r,r_s,p_s,G=4302./10.**9):
    C1 = -4*pi*G*p_s*r_s**3
    C2 = r_s
    C3 = p_s*r_s
    return C3*(-C1*np.log(1 + r/C2)/r**2 + C1/(C2*r*(1 + r/C2)))/(r*(1 + r/C2)**2)

def vel_disp_NFW(r,r_s,p_s,G=4302./10.**9):
    integral = quad(jeans_integrand, r, math.inf, args=(r_s,p_s))[0]
    return np.sqrt(integral/p_NFW(r,r_s,p_s))
    
# Constants
#------------------------------------------------------------------------------
pi = np.pi
G=4302./10.**9
mpc_to_kpc = 1/1000
H_0 = 70 * mpc_to_kpc
p_crit = 3 * (H_0)**2 * (8*np.pi*G)**(-1)
a=(16/pi)**(1/2)
b=(25*(pi)**(1/2))/32
C=290/385
sigma_convert = (1/3.086e+21)**2 * (1.989e+33/1) # converts cm to kpc, grams to M_sun
kpc_per_km = 1/(3.086e+16)

# Comparing vel_disp codes
#------------------------------------------------------------------------------
#p_s = 1.28e+7
#r_s = 6.5
#r_vals = np.logspace(-2,3,50)*r_s
#plt.xscale('log')
#plt.plot(r_vals,[vel_disp_NFW_Laura(r,r_s,p_s) for r in r_vals],'.')
#plt.plot(r_vals,np.array([vel_disp_NFW(r,r_s,p_s) for r in r_vals])+0.25,'.')
#plt.show()

# Initialize
#------------------------------------------------------------------------------
p_s = 1.28e+7 # Essig pg. 2
r_s = 6.5 # Essig pg. 2
sigma = 3*sigma_convert # Essig pg. 2
N = 150
r_vals = np.logspace(-2,3,N)*r_s
M_vals = np.zeros(N)
p_vals = np.zeros(N)
nu_vals = np.zeros(N)
u_vals = np.zeros(N)
for i in range (N):
    M_vals[i] = M_NFW(r_vals[i],r_s,p_s)
    p_vals[i] = p_NFW (r_vals[i],r_s,p_s)
    nu_vals[i]= vel_disp_NFW(r_vals[i],r_s,p_s)
    u_vals[i] = (3/2)*nu_vals[i]**2
    
# Update luminosity (step 1)
#------------------------------------------------------------------------------
deriv_vals = np.zeros(N)
L_vals = np.zeros(N)
for i in range (N):
    if i==0:
        deriv_vals[i] = ( (nu_vals[i+1])**2 - (nu_vals[i])**2 )/ (r_vals[i+1] - r_vals[i])
    elif i==N-1:
        deriv_vals[i] = ( (nu_vals[i])**2 - (nu_vals[i-1])**2 )/ (r_vals[i] - r_vals[i-1])
    else:
        deriv_vals[i] = ( (nu_vals[i+1])**2 - (nu_vals[i-1])**2 )/ (r_vals[i+1] - r_vals[i-1])
    factor = ( a*sigma**2 + (4*pi*G*b)/(p_vals[i]*nu_vals[i]**2*C) )**(-1)
    L_vals[i]=(4*pi*r_vals[i]**2)*(-3/2)*a*b*nu_vals[i]*sigma*factor*deriv_vals[i]

LHS_vals = np.zeros(N)
RHS_vals = np.zeros(N)
for i in range (N):
    if i==0:
        LHS_vals[i] = ( p_vals[i+1]*nu_vals[i+1]**2 - p_vals[i]*nu_vals[i]**2 )/ (r_vals[i+1] - r_vals[i])
    elif i==N-1:
        LHS_vals[i] = ( p_vals[i]*nu_vals[i]**2 - p_vals[i-1]*nu_vals[i-1]**2 )/ (r_vals[i] - r_vals[i-1])
    else:
        LHS_vals[i] = ( p_vals[i+1]*nu_vals[i+1]**2 - p_vals[i-1]*nu_vals[i-1]**2 )/ (r_vals[i+1] - r_vals[i-1])
    RHS_vals[i] = -G*M_vals[i]*p_vals[i]*(r_vals[i])**(-2)
plt.plot(range(N),np.abs((LHS_vals-RHS_vals)/LHS_vals),'.',color='blue')
    
# Time evolution (step 2)
#------------------------------------------------------------------------------
inequality = 0
delta_t = 10**15
delta_u_vals = np.zeros(N)
deriv_vals = np.zeros(N)
u_vals_temp = copy.deepcopy(u_vals)
while inequality == 0:
    for i in range (N):
        if i==0:
            deriv_vals[i] = ( L_vals[i+1] - L_vals[i] )/ (M_vals[i+1] - M_vals[i])
        elif i==N-1:
            deriv_vals[i] = ( L_vals[i] - L_vals[i-1] )/ (M_vals[i] - M_vals[i-1])
        else:
            deriv_vals[i] = ( L_vals[i+1] - L_vals[i-1] )/ (M_vals[i+1] - M_vals[i-1])
        delta_u_vals[i] = -deriv_vals[i] * delta_t * kpc_per_km
        u_vals[i] = u_vals[i] + delta_u_vals[i]
        nu_vals[i] = np.sqrt( (2/3)*u_vals[i] )
        if np.abs( delta_u_vals[i]/u_vals[i] ) > 1e-3:
            delta_t = delta_t/10
            delta_u_vals = np.zeros(N)
            deriv_vals = np.zeros(N)
            u_vals=copy.deepcopy(u_vals_temp)
            break
        if i==N-1:
            inequality=1

for i in range (N):
    if i==0:
        LHS_vals[i] = ( p_vals[i+1]*nu_vals[i+1]**2 - p_vals[i]*nu_vals[i]**2 )/ (r_vals[i+1] - r_vals[i])
    elif i==N-1:
        LHS_vals[i] = ( p_vals[i]*nu_vals[i]**2 - p_vals[i-1]*nu_vals[i-1]**2 )/ (r_vals[i] - r_vals[i-1])
    else:
        LHS_vals[i] = ( p_vals[i+1]*nu_vals[i+1]**2 - p_vals[i-1]*nu_vals[i-1]**2 )/ (r_vals[i+1] - r_vals[i-1])
    RHS_vals[i] = -G*M_vals[i]*p_vals[i]*(r_vals[i])**(-2)
plt.plot(range(N),np.abs((LHS_vals-RHS_vals)/LHS_vals),'.',color='green')
plt.show()