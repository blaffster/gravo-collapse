# -*- coding: utf-8 -*-
"""
Created on Thu May 28 08:27:22 2020

@author: nebue
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.integrate import odeint


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
def nu_integrand(r,r_s,rho_s):
    return M_NFW(r,r_s,rho_s) * rho_NFW (r,r_s,rho_s) * r**(-2)


# Function to get P_NFW
#------------------------------------------------------------------------------
def nu_NFW(r,r_s,rho_s,G = 4.302e-06):
    integral = quad(nu_integrand, r, math.inf, args=(r_s,rho_s))[0]
    return (G/rho_NFW (r,r_s,rho_s)) * integral


# Function to calculate finite differences
#------------------------------------------------------------------------------
def deriv(y1,y2,Delta_r):
    return (y2-y1)/Delta_r


# Function to time-evolve densities
#------------------------------------------------------------------------------
def rho_deriv_RHS(N,rho,u,Delta_r,r):
    vals = np.zeros(N)
    for i in range(N):
        if i<N-1:
            vals[i] = -rho[i]*deriv(u[i],u[i+1],Delta_r[i]) - u[i]*deriv(rho[i],rho[i+1],Delta_r[i]) - (2/r[i])*rho[i]*u[i]
        else:
            vals[i] = -rho[i]*deriv(u[i],0,Delta_r[i]) - u[i]*deriv(rho[i],0,Delta_r[i]) - (2/r[i])*rho[i]*u[i]
    return vals


# Function to time-evolve fluid velocities
#------------------------------------------------------------------------------
def u_deriv_RHS(N,u,Delta_r,M,r,rho,p,G = 4.302e-06):
    vals = np.zeros(N)
    for i in range(N):
        if i<N-1:
            vals[i] = -u[i]*deriv(u[i],u[i+1],Delta_r[i]) - (G*M[i])/r[i]**2 - (1/rho[i])*deriv(p[i],p[i+1],Delta_r[i])
        else:
            vals[i] = -u[i]*deriv(u[i],0,Delta_r[i]) - (G*M[i])/r[i]**2 - (1/rho[i])*deriv(p[i],0,Delta_r[i])
    return vals


# Function to time-evolve 1D vel. disp.
#------------------------------------------------------------------------------
def nu_deriv_RHS(N,nu,r,p,L,Delta_r,u):
    vals = np.zeros(N)
    for i in range(N):
        if i<N-1:
            vals[i] = (-1/(4*np.pi*r[i]**2*p[i]))*deriv(L[i],L[i+1],Delta_r[i]) - (deriv(u[i],u[i+1],Delta_r[i]) + (2*u[i])/r[i]) - (3*u[i])/(nu[i])*deriv(nu[i],nu[i+1],Delta_r[i])
        else:
            vals[i] = (-1/(4*np.pi*r[i]**2*p[i]))*deriv(L[i],0,Delta_r[i]) - (deriv(u[i],0,Delta_r[i]) + (2*u[i])/r[i]) - (3*u[i])/(nu[i])*deriv(nu[i],0,Delta_r[i])         
    return vals

# Function to update mass
#------------------------------------------------------------------------------
def M_new(N,r,Delta_r,rho):
    SUM = 0
    M=np.zeros(N)
    for i in range(N):
        if i==0:
            SUM+=(4*np.pi)/(3) * Delta_r[0]**3 * rho[0]
            M[0]=SUM
        else:
            SUM+= 4 * np.pi * r[i]**2 * rho[i] * Delta_r[i]
            M[i] = SUM
    return M


# Function to update luminosity
#------------------------------------------------------------------------------
def L_new(N,r,nu,p,sigma,Delta_r,G = 4.302e-06,a = (16/np.pi)**(1/2),b = (25 * (np.pi)**(1/2) )/32,C = 290/385):
    L = np.zeros(N)
    for i in range(N-1):
        term_1 = -12 * np.pi * r[i]**2 * nu[i]**2 * a * b * sigma
        term_2 = ( a*sigma**2 + (b/C) * (4*np.pi*G)/(p[i]) )**(-1)
        term_3 = deriv(nu[i],nu[i+1],Delta_r[i])
        L[i] = term_1 * term_2 * term_3
    return L


# Function to update pressure
#------------------------------------------------------------------------------
def p_new(N,rho,nu):
    p = np.zeros(N)
    for i in range (N):
        p[i] = rho[i] * nu[i]**2
    return p


# odeint functions
#------------------------------------------------------------------------------
def dy_dt(y,t,N,r,Delta_r):
    return RHS(y,N,r,Delta_r)


def RHS(y,N,r,Delta_r):
    rho = y[:N]
    u = y[N:2*N]
    nu = y[2*N:3*N]
    
    M = M_new(N,r,Delta_r,rho)
    p = p_new(N,rho,nu)
    L = L_new(N,r,nu,p,sigma,Delta_r)
    
    drho_dt_RHS = rho_deriv_RHS(N,rho,u,Delta_r,r)
    du_dt_RHS = u_deriv_RHS(N,u,Delta_r,M,r,rho,p)
    dnu_dt_RHS = nu_deriv_RHS(N,nu,r,p,L,Delta_r,u)
    
    return np.concatenate((drho_dt_RHS,du_dt_RHS,dnu_dt_RHS))



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
N = 150 # Nishikawa pg. 5
s = np.logspace(-2,3,N+1) * r_s # Essig pg. 7
r = []
Delta_r = []
for i in range(N):
    r.append( (s[i] + s[i+1])/2 )
    Delta_r.append(s[i+1] - s[i])


rho = [rho_NFW (x,r_s,rho_s) for x in r]
u = np.zeros(N)
nu = [nu_NFW (x,r_s,rho_s) for x in r]
    

# Time evolve quantities
#------------------------------------------------------------------------------
y_0 = np.concatenate((rho,u,nu))
t = np.linspace(0,0.1,10)
y = odeint(dy_dt,y_0,t,args=(N,r,Delta_r))

print(y[-1][:N]) # final rho values
print(y[-1][N:2*N]) # final u values
print(y[-1][2*N:3*N]) # final nu values



# Plotting
#------------------------------------------------------------------------------
#plt.clf()
#plt.loglog(r,rho,label='Density')
#plt.loglog(r,M,label='Mass')
#plt.loglog(r,p,label='Pressure')
#plt.loglog(r,nu,label='1D Vel. Disp.')
#plt.loglog(r,np.abs(L),label='Luminosity (abs val)')
#plt.xlabel('r (kpc)')
#plt.legend()
#plt.show()