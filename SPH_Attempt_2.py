# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 09:54:14 2020

@author: nebue
"""

import numpy as np
import math
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
pi = np.pi
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}
plt.rc('font', **font)


# Function to get rho_NFW
#------------------------------------------------------------------------------
def rho_NFW(r,r_s,rho_s):
    X = r/r_s
    return rho_s * (X)**(-1) * (1 + X)**(-2)


# Function to get M_NFW
#------------------------------------------------------------------------------
def M_NFW(r,r_s,rho_s):
    X = r/r_s
    if X > 1e-5:
        return 4*np.pi*rho_s*r_s**3*( np.log(1 + X) - (X)/(1 + X) )
    else:
        return 4*np.pi*rho_s*r_s**3*((1/2)*X**2 - (2/3)*X**3 + (3/4)*X**4 - (4/5)*X**5)


# Function to get P_NFW_integrand
#------------------------------------------------------------------------------
def P_NFW_integrand(r,r_s,rho_s,G = 4.302e-06):
    return G * M_NFW(r,r_s,rho_s) * rho_NFW (r,r_s,rho_s) * r**(-2)


# Function to get P_NFW
#------------------------------------------------------------------------------
def P_NFW(r,r_s,rho_s):
    integral = quad(P_NFW_integrand, r, math.inf, args=(r_s,rho_s))[0]
    return integral


# Functions to calculate C(q) and D(q)
#------------------------------------------------------------------------------
def C(q):
    return q**2 - (3/4)*q**4 + (3/10)*q**5

def D(q):
    return 2*q**2 - 2*q**3 + (3/4)*q**4 - (1/10)*q**5


# Function to calculate W_3S1
#------------------------------------------------------------------------------
def W_3S1(rp,r,h):
    σ = r/h
    σ_prime = rp/h
    X = σ_prime + σ
    Y = abs(σ_prime - σ)
    prefactor = 1/(h*rp*r)
    
    if 2 < X:
        if 0 <= Y < 1:
            return prefactor * ( (7/10) - C(Y) )
        elif 1 <= Y < 2:
            return prefactor * ( (8/10) - D(Y) )
        else:
            return 0
        
    elif 1 < X <= 2:
        if 0 <= Y < 1:
            return prefactor * ( (-1/10) + D(X) - C(Y) )
        elif 1 <= Y < 2:
            return prefactor * ( D(X) - D(Y) )
        else:
            return 0
        
    elif X <= 1:
        return prefactor * ( C(X) - C(Y) )
    

# Function to calculate dW/dr
#------------------------------------------------------------------------------   
def dWdr(rp,r,h):
    σ = r/h
    σ_prime = rp/h
    X = σ_prime + σ
    Y = abs(σ_prime - σ)
    Z = σ_prime - σ
    
    if X <= 1:
        if Z > 0:
            t1 = 12*r
            t2 = r**2 + 5*rp*(-h + rp)
            t3 = 5 * h**6 * rp
            return (t1*t2)/(t3)
        else:
            t1 = 20*h*r**3 - 15*r**4 - 10*r**2*rp**2 + rp**4
            t2 = 5 * h**6 * r**2
            return (t1/t2) * (-3)
        
    elif 1 < X <= 2:
        if 0 <= Y < 1:
            if Z > 0:
                t1 = h**5 - 20*h**2*(2*r - rp)*(r + rp)**2
                t2 = 10*h**3*(r**2 - rp**2) + 15*h*(3*r**4 + 6*r**2*rp**2 - rp**4)
                t3 = 4*(2*r**5 - 15*r**4*rp + 10*r**3*rp**2 - 10*r**2*rp**3 + rp**5)
                t4 = 10 * h**6 * r**2 * rp
                return (t1 + t2 + t3)/(t4)
            else:
                t1 = h**5 - 20*h**2*(2*r - rp)*(r + rp)**2
                t2 = 10*h**3*(r**2 - rp**2) + 15*h*(3*r**4 + 6*r**2*rp**2 - rp**4)
                t3 = -2*(8*r**5 - 15*r**4*rp + 40*r**3*rp**2 - 10*r**2*rp**3 + rp**5)
                t4 = 10 * h**6 * r**2 * rp
                return (t1 + t2 + t3)/(t4)
        elif 1 <= Y < 2:
            if Z > 0:
                t1 = 4*r
                t2 = 10*h**2 + r**2 - 15*h*rp + 5*rp**2
                t3 = 5 * h**6 * rp
                return -(t1*t2)/(t3)
            else:
                t1 = 60*h*r**3 - 15*r**4 - 10*r**2*rp**2
                t2 = rp**4 + 20*h**2*(-3*r**2 + rp**2)
                t3 = 5 * h**6 * r**2
                return (t1 + t2)/(t3)
        else:
            return 0
    
    elif 2 < X:
        if 0 <= Y < 1:
            if Z > 0:
                t1 = -14*h**5 + 15*h*(r - rp)**3*(3*r + rp)
                t2 = 6*(r - rp)**4*(4*r + rp) - 20*h**3*(r**2 - rp**2)
                t3 = 20 * h**6 * r**2 * rp
                return (t1 + t2)/(t3)
            else:
                t1 = 14*h**5 - 15*h*(r - rp)**3*(3*r + rp)
                t2 = 6*(r - rp)**4*(4*r + rp) + 20*h**3*(r**2 - rp**2)
                t3 = 20 * h**6 * r**2 * rp
                return -(t1 + t2)/(t3)
        elif 1 <= Y < 2:
            if Z > 0:
                t1 = (2*h + r - rp)**3
                t2 = 2*h**2 - 3*h*r + 8*r**2 + 3*h*rp - 6*r*rp - 2*rp**2
                t3 = 20 * h**6 * r**2 * rp
                return -(t1*t2)/(t3)
            else:
                t1 = (2*h - r + rp)**3
                t2 = 2*h**2 + 8*r**2 + 3*h*(r - rp) - 6*r*rp - 2*rp**2
                t3 = 20 * h**6 * r**2 * rp
                return -(t1*t2)/(t3)
        else:
            return 0


# Function to calculate dW/drp
#------------------------------------------------------------------------------
def dWdrp(rp,r,h):
    σ = r/h
    σ_prime = rp/h
    X = σ_prime + σ
    Y = abs(σ_prime - σ)
    Z = σ_prime - σ
    
    if X <= 1:
        if Z > 0:
            t1 = r**4 - 10*r**2*rp**2 + 5*(4*h - 3*rp)*rp**3
            t2 = 5 * h**6 * rp**2
            return (-3) * (t1/t2)
        else:
            t1 = 12*rp
            t2 = -5*h*r + 5*r**2 + rp**2
            t3 = 5 * h**6 * r
            return (t1*t2)/(t3)

    elif 1 < X <= 2:
        if 0 <= Y < 1:
            if Z > 0:
                t1 = h**5 + 20*h**2*(r - 2*rp)*(r + rp)**2
                t2 = -10*h**3*(r**2 - rp**2) - 15*h*(r**4 - 6*r**2*rp**2 - 3*rp**4)
                t3 = -2*(r**5 - 10*r**3*rp**2 + 40*r**2*rp**3 - 15*r*rp**4 + 8*rp**5)
                t4 = 10 * h**6 * r * rp**2
                return (t1 + t2 + t3)/(t4)
            else:
                t1 = h**5 + 20*h**2*(r - 2*rp)*(r + rp)**2
                t2 = -10*h**3*(r**2 - rp**2) - 15*h*(r**4 - 6*r**2*rp**2 - 3*rp**4)
                t3 = 4*(r**5 - 10*r**3*rp**2 + 10*r**2*rp**3 - 15*r*rp**4 + 2*rp**5)
                t4 = 10 * h**6 * r * rp**2
                return (t1 + t2 + t3)/(t4)
        elif 1 <= Y < 2:
            if Z > 0:
                t1 = r**4 - 10*r**2*rp**2 + 60*h*rp**3
                t2 = -15*rp**4 + 20*h**2*(r**2 - 3*rp**2)
                t3 = 5 * h**6 * rp**2
                return (t1 + t2)/(t3)
            else:
                t1 = 4*rp
                t2 = 10*h**2 - 15*h*r + 5*r**2 + rp**2
                t3 = 5 * h**6 * r
                return -(t1*t2)/(t3)
        else:
            return 0
    
    elif 2 < X:
        if 0 <= Y < 1:
            if Z > 0:
                t1 = 14*h**5 + 15*h*(r - rp)**3*(r + 3*rp)
                t2 = 6*(r - rp)**4*(r + 4*rp) - 20*h**3*(r**2 - rp**2)
                t3 = 20 * h**6 * r * rp**2
                return -(t1 + t2)/(t3)
            else:
                t1 = -14*h**5 - 15*h*(r - rp)**3*(r + 3*rp)
                t2 = 6*(r - rp)**4*(r + 4*rp) + 20*h**3*(r**2 - rp**2)
                t3 = 20 * h**6 * r * rp**2
                return (t1 + t2)/(t3)
        elif 1 <= Y < 2:
            if Z > 0:
                t1 = (2*h + r - rp)**3
                t2 = 2*h**2 - 3*h*r - 2*r**2 + 3*h*rp - 6*r*rp + 8*rp**2
                t3 = 20 * h**6 * r * rp**2
                return -(t1*t2)/(t3)
            else:
                t1 = (2*h - r + rp)**3
                t2 = 2*h**2 + 3*h*r - 2*r**2 - 3*h*rp - 6*r*rp + 8*rp**2
                t3 = 20 * h**6 * r * rp**2
                return -(t1*t2)/(t3)
        else:
            return 0


# Function to get p_s
#------------------------------------------------------------------------------
def get_rho_s(c,p_crit):
    return (200*p_crit/3) * (c**3) * (np.log(1 + c) - c/(1 + c))**(-1)


# Function to get r_s
#------------------------------------------------------------------------------
def get_r_s(M_200,c,p_s):
    return (M_200 * (4*np.pi*p_s)**(-1) * (np.log(1 + c) - c/(1 + c))**(-1))**(1/3)


# Function used to determine c from rho_s
#------------------------------------------------------------------------------
def c_func(c,rho_s,p_crit):
    return rho_s - (200*p_crit/3) * (c**3) * (np.log(1 + c) - c/(1 + c))**(-1)


# Function used to determine r from M(r)
#------------------------------------------------------------------------------
def r_func(r,M,r_s,rho_s):
    X = r/r_s
    return M - 4*np.pi*rho_s*r_s**3*( np.log(1 + X) - (X)/(1 + X) )


# Function used to determine rho_i
#------------------------------------------------------------------------------
def get_rho(N,i,r,h,m):
    rho = 0
    for j in range(0,N):
        rho += m * W_3S1(r[j],r[i],h[i])
    return rho


# Function used to determine dvdt
#------------------------------------------------------------------------------
def dvdt(N,i,m,P,rho,r,h,G=4.302e-6):
    RHS = 0
    for j in range(0,N):
        if j != i:
            term1 = -m * ( (P[i]/rho[i]**2) + (P[j]/rho[j]**2) ) * dWdr(r[j],r[i],h[i])
            if r[i] > r[j]:
                term2 = -G * (m/r[j]**2)
            else:
                term2 = 0
            RHS += term1 + term2
    return RHS


# Function used to determine dudt
#------------------------------------------------------------------------------
def dedt(N,i,m,P,rho,v,h,r):
    RHS = 0
    for j in range(0,N):
        if j != i:
            RHS += m * (P[i]/rho[i]**2) * ( v[j]*dWdrp(r[j],r[i],h[i]) + v[i]*dWdr(r[j],r[i],h[i]) )
    return RHS


# Integrand for mass integral
#------------------------------------------------------------------------------
#def M_integrand(rp,r,h):
#    σ = r/h
#    σ_prime = rp/h
#    X = σ_prime + σ
#    Y = abs(σ_prime - σ)
#    prefactor = 1/(h*rp*r)
#    
#    if 2 < X:
#        if 0 <= Y < 1:
#            return prefactor * ( (7/10) - C(Y) ) * rp**2
#        elif 1 <= Y < 2:
#            return prefactor * ( (8/10) - D(Y) ) * rp**2
#        else:
#            return 0
#        
#    elif 1 < X <= 2:
#        if 0 <= Y < 1:
#            return prefactor * ( (-1/10) + D(X) - C(Y) ) * rp**2
#        elif 1 <= Y < 2:
#            return prefactor * ( D(X) - D(Y) ) * rp**2
#        else:
#            return 0
#        
#    elif X <= 1:
#        return prefactor * ( C(X) - C(Y) ) * rp**2


# Initialize constants
#------------------------------------------------------------------------------
rho_s = 0.019 * (1000/1)**3 # M_sun/kpc^3, Nishikawa pg.9
r_s = 2.59 # kpc, Nishikawa pg.9
G = 4.302e-6
mpc_to_kpc = 1/1000
H_0 = 70 * mpc_to_kpc
p_crit = 3 * (H_0)**2 * (8*np.pi*G)**(-1)


# Find virial radius/concentration parameter
#------------------------------------------------------------------------------
c = fsolve(c_func, 1, args=(rho_s,p_crit))[0]
R_vir = c*r_s


# Set number of particles/particle mass
#------------------------------------------------------------------------------
M_vir = M_NFW(R_vir,r_s,rho_s)
N = 500
m = 1/(4*pi) * (M_vir/N)


# Sample initial density distribution, initialize r_i/v_i/P_i
#------------------------------------------------------------------------------
M_samples = np.zeros(N)
np.random.seed(0)
for i in range(0,N):
    M_samples[i] = np.random.uniform()*M_vir
M_samples = np.sort(M_samples)
steps = 20
r = np.zeros((steps+1,N))
v = np.zeros((steps+1,N))
P = np.zeros((steps+1,N))
for j in range(0,N):
    r[0,j] = fsolve(r_func, 1, args=(M_samples[j],r_s,rho_s))[0]
    P[0,j] = P_NFW(r[0,j],r_s,rho_s)

# Initialize smoothing lengths h_i
#------------------------------------------------------------------------------
eta = 1.5
h = np.zeros((steps+1,N))
for j in range(N):
    h[0,j] = eta * (m/rho_NFW(r[0,j],r_s,rho_s))**(1/3)
    
    
# Calculate/plot SPH mass profile, compare it to NFW
#------------------------------------------------------------------------------
#n = 75
#r_samp = np.linspace(r[0],r[-1],n)
#M = np.zeros(n)
#for i in range(n):
#    for j in range(N):
#        M[i] += 4*pi*m*quad(M_integrand, 0, r_samp[i], args=(r[j],h[j]))[0]
#plt.clf()
#plt.plot(r_samp,[M_NFW(x,r_s,rho_s) for x in r_samp],label='NFW')
#plt.plot(r_samp,M,'.',label='SPH')s
#plt.title('Mass profiles NFW vs. SPH (N = 5000, n = 75, η = 1.5)')
#plt.xlabel('r (kpc)')
#plt.ylabel('M (M_sun)')
#plt.legend()
#plt.show()


# Calculate/plot SPH density profile, compare it to NFW, initialize u
#------------------------------------------------------------------------------
rho = np.zeros((steps+1,N))
for j in range(0,N):
    rho[0,j] = get_rho(N,j,r[0],h[0],m)
e = np.zeros((steps+1,N))
e[0] = (3/2)*(P[0]/rho[0])
#plt.clf()
#plt.loglog(r,[rho_NFW(x,r_s,rho_s) for x in r],label='NFW')
#plt.loglog(r,rho,'.',label='SPH')
#plt.xlabel('r  [kpc]')
#plt.ylabel('ρ(r)  [M_sun/kpc^3]')
#plt.title('Density profiles NFW vs. SPH (N = 5000, η = 1.5)')
#plt.legend()
#plt.show()


# Begin time-evolution
#------------------------------------------------------------------------------
#f_F = (0.25+0.5)/2
#dt_F = np.zeros(N)
dt = 0.001
t_elapsed = 0
print('t_elapsed =',t_elapsed,'  r_0 =', r[0,0],'  rho_0 =',rho[0,0])
for i in range (1,steps+1):
#    for j in range(0,N):
#        a_abs = np.abs ( dvdt(N,j,m,P[i-1],rho[i-1],r[i-1],h[i-1]) )
#        dt_F[j] = f_F*np.sqrt(h[i-1,j]/a_abs)
#    dt = min(dt_F)
#    print(dt)
    for j in range(0,N):
        # Update r, v, and u
        v[i,j] = v[i-1,j] + dt*dvdt(N,j,m,P[i-1],rho[i-1],r[i-1],h[i-1])
        r[i,j] = r[i-1,j] + dt*v[i-1,j]
        e[i,j] = e[i-1,j] + dt*dedt(N,j,m,P[i-1],rho[i-1],v[i-1],h[i-1],r[i-1])
#        print(r[i,j])
    for j in range(0,N):
        # Update rho, P, and h
        rho[i,j] = get_rho(N,j,r[i],h[i-1],m)
        P[i,j] = (2/3) * rho[i,j] * e[i,j]
        h[i,j] = eta * (m/rho[i,j])**(1/3)
    t_elapsed += dt
    print('t_elapsed =',t_elapsed,'  r_0 =', r[i,0],'  rho_0 =',rho[i,0])
print(min(r[-1]))

plt.clf()
plt.title('SPH Density Profile Evolution (N=1000,steps=15,dt=0.001)')
plt.xlabel('r  [kpc]')
plt.ylabel('ρ(r)  [M_sun/kpc^3]')
r_vals = np.linspace(min(r[-1]),max(r[-1]),1000)
plt.loglog(r_vals,[rho_NFW(x,r_s,rho_s) for x in r_vals],label='NFW')
plt.loglog(r[0],rho[0],linestyle='--', marker='o',markersize=3,label='Initial SPH')
plt.loglog(r[-1],rho[-1],linestyle='--', marker='o',markersize=3,label='Final SPH')
plt.legend()
plt.show()