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
def nu_integrand(r,r_s,rho_s,G = 4.302e-06):
    return G * M_NFW(r,r_s,rho_s) * rho_NFW (r,r_s,rho_s) * r**(-2)


# Function to get P_NFW
#------------------------------------------------------------------------------
def nu_NFW(r,r_s,rho_s):
    integral = quad(nu_integrand, r, math.inf, args=(r_s,rho_s))[0]
    return np.sqrt( (1/rho_NFW (r,r_s,rho_s)) * integral )


# Function to calculate finite differences
#------------------------------------------------------------------------------
def deriv(y1,y2,Delta_r):
    return (y2-y1)/Delta_r


# Function to time-evolve densities
#------------------------------------------------------------------------------
def rho_deriv_RHS(N,rho,u,Delta_r,r):
    vals = np.zeros(N)
    for i in range(N):
        if i!=N-1:
            vals[i] = -rho[i]*deriv(u[i],u[i+1],Delta_r[i]) - u[i]*deriv(rho[i],rho[i+1],Delta_r[i]) - (2/r[i])*rho[i]*u[i]
        else:
            vals[i] = -rho[i]*deriv(u[i],0,Delta_r[i]) - u[i]*deriv(rho[i],0,Delta_r[i]) - (2/r[i])*rho[i]*u[i]
    return vals


# Function to time-evolve fluid velocities
#------------------------------------------------------------------------------
def u_deriv_RHS(N,u,Delta_r,M,r,rho,p,G = 4.302e-06):
    vals = np.zeros(N)
    for i in range(N):
        if i!=N-1:
            vals[i] = -u[i]*deriv(u[i],u[i+1],Delta_r[i]) - (G*M[i])/r[i]**2 - (1/rho[i])*deriv(p[i],p[i+1],Delta_r[i])
        else:
            vals[i] = -u[i]*deriv(u[i],0,Delta_r[i]) - (G*M[i])/r[i]**2 - (1/rho[i])*deriv(p[i],0,Delta_r[i])
    return vals


# Function to time-evolve 1D vel. disp.
#------------------------------------------------------------------------------
def nu_deriv_RHS(N,nu,r,p,L,Delta_r,u):
    vals = np.zeros(N)
    for i in range(N):
        if i!=N-1:
            vals[i] = (-nu[i]/3) * ( 1/(4*np.pi*r[i]**2*p[i])*deriv(L[i],L[i+1],Delta_r[i]) + deriv(u[i],u[i+1],Delta_r[i]) + (2*u[i])/r[i] + (3*u[i])/(nu[i])*deriv(nu[i],nu[i+1],Delta_r[i]) )
        else:
            vals[i] = (-nu[i]/3) * ( 1/(4*np.pi*r[i]**2*p[i])*deriv(L[i],0,Delta_r[i]) + deriv(u[i],0,Delta_r[i]) + (2*u[i])/r[i] + (3*u[i])/(nu[i])*deriv(nu[i],0,Delta_r[i]) )        
    return vals


# Function to update mass
#------------------------------------------------------------------------------
def M_new(N,r,s,rho):
    M = np.zeros(N)
    M[0] = (4*np.pi)/(3) * rho[0] * r[0]**3
    for i in range(1,N):
        M[i] = (4*np.pi)/(3)*rho[0]*s[1]**3 + (4*np.pi)/(3)*rho[i]*( r[i]**3 - s[i]**3 )
        for j in range(1,i):
                M[i]+= (4*np.pi)/(3) * rho[j] * ( s[j+1]**3 - s[j]**3 )
    return M


# Function to update luminosity
#------------------------------------------------------------------------------
def L_new(N,r,nu,p,sigma,Delta_r,G = 4.302e-06,a=(16/np.pi)**(1/2),b=(25 * (np.pi)**(1/2))/32,C=290/385):
    L = np.zeros(N)
    for i in range(N):
        term_1 = -12 * np.pi * r[i]**2 * nu[i]**2 * a * b * sigma
        term_2 = ( a*sigma**2 + (b/C)*(4*np.pi*G)/(p[i]) )**(-1)
        if i!=N-1:
            term_3 = deriv(nu[i],nu[i+1],Delta_r[i])
        else:
            term_3 = deriv(nu[i],0,Delta_r[i])   
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
def RHS(y,N,r,Delta_r,sigma):
    rho = y[:N]
    u = y[N:2*N]
    nu = y[2*N:3*N]
    
    M = M_new(N,r,s,rho)
    p = p_new(N,rho,nu)
    L = L_new(N,r,nu,p,sigma,Delta_r)
    
    drho_dt_RHS = rho_deriv_RHS(N,rho,u,Delta_r,r)
    du_dt_RHS = u_deriv_RHS(N,u,Delta_r,M,r,rho,p)
    dnu_dt_RHS = nu_deriv_RHS(N,nu,r,p,L,Delta_r,u)
    
    return np.concatenate((drho_dt_RHS,du_dt_RHS,dnu_dt_RHS))


def dy_dt(y,t,N,r,Delta_r,sigma):
    return RHS(y,N,r,Delta_r,sigma)


# Initialize constants and choose rho_s, r_s, and sigma_m
#------------------------------------------------------------------------------
G = 4.302e-06 # (kpc/M_sun)(km/s)^2
a = (16/np.pi)**(1/2) # Nishikawa pg.1
b = (25 * (np.pi)**(1/2) )/32 # Nishikawa pg.1
C = 0.75 # Nishikawa pg.2


rho_s = 0.019 * (1000/1)**3 # M_sun/kpc^3, Nishikawa pg.9
r_s = 2.59 # kpc, Nishikawa pg.9
sigma = 5 # cm^2/g, Nishikawa pg.9
sigma_convert = (1/3.086e+21)**2 * (1.989e+33/1) # converts cm^2 to kpc^2, grams^-1 to M_sun^-1
sigma = sigma*sigma_convert # kpc^2/M_sun


# Discretize halo and initialize quantities
#------------------------------------------------------------------------------
N = 100
s = np.logspace(-5,2,N) * r_s
s = np.insert(s,0,0)
r = []
Delta_r = []
for i in range(N):
    r.append( (s[i] + s[i+1])/2 )
    Delta_r.append(s[i+1] - s[i])


rho = [rho_NFW (x,r_s,rho_s) for x in r]
u = np.zeros(N)
nu = [nu_NFW (x,r_s,rho_s) for x in r]
#nu = np.ones(N) * np.mean(nu)

#M1 = [M_NFW(x,r_s,rho_s) for x in r]
#M2 = M_new(N,r,s,rho)
#plt.clf()
#plt.loglog(r,M1)
#plt.loglog(r,M2)
#plt.show()


# Time evolve quantities
#------------------------------------------------------------------------------
#y_0 = np.concatenate((rho,u,nu))
#duration = 0.1 # Gyr
#points = 20
#t = np.linspace(0,duration,points)
#y = odeint(dy_dt,y_0,t,args=(N,r,Delta_r,sigma))
#
#plt.clf()
#for i in range(points):
#    print(y[i][2*N:3*N])
#    plt.loglog(r,y[i][2*N:3*N])
#plt.show()


M = M_new(N,r,s,rho)
p = p_new(N,rho,nu)
#L = L_new(N,r,nu,p,sigma,Delta_r)
#drho_dt_RHS = rho_deriv_RHS(N,rho,u,Delta_r,r)
#du_dt_RHS = u_deriv_RHS(N,u,Delta_r,M,r,rho,p)
#dnu_dt_RHS = nu_deriv_RHS(N,nu,r,p,L,Delta_r,u)
#print(drho_dt_RHS,'\n')
#print(du_dt_RHS,'\n')
#print(dnu_dt_RHS,'\n')
vals = np.zeros(N)
for i in range(N):
    if i!=N-1:
        vals[i] = -u[i]*deriv(u[i],u[i+1],Delta_r[i]) - (G*M[i])/r[i]**2 - (1/rho[i])*deriv(p[i],p[i+1],Delta_r[i])
    else:
        vals[i] = -u[i]*deriv(u[i],0,Delta_r[i]) - (G*M[i])/r[i]**2 - (1/rho[i])*deriv(p[i],0,Delta_r[i])
print(vals)