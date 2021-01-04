# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 09:54:14 2020

@author: nebue
"""

#import pandas as pd
#y = pd.read_csv('neighbors.csv')
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
def get_rho(N,i,index,r,h,m):
    rho_ij = 0
    for j in index:
        rho_ij += (m * (4*pi)**(-1)) * W_3S1(r[j],r[i],(h[i]+h[j])/2)
    return rho_ij


# Function used to determine dvdt
#------------------------------------------------------------------------------
def dvdt(N,i,index,m,P,rho,r,h,G=4.302e-6):
    RHS = 0
    force = 0
    pressure = 0
    for j in index:
        if j != i:
            term1 = rho[i] * (-m * (4*pi)**(-1)) * ( (P[i]/rho[i]**2) + (P[j]/rho[j]**2) ) * dWdr(r[j],r[i],h[i])
#            term1 = rho[i] * (-m * (4*pi)**(-1)) * ( (P[i]/rho[i]**2)*dWdr(r[j],r[i],h[i]) + (P[j]/rho[j]**2)*dWdr(r[j],r[i],h[j]) )
            pressure += term1
            RHS += term1
    r_new = np.sort(r)
    i_new = np.where(r_new-r[i] == 0)[0][0]
    term2 = -(G*m)/r[i]**2
    for j in range(i_new+1):
#        if j<i_new:
        force += term2
        RHS += term2
#        else:
#            force += (1/2)*term2
#            RHS += (1/2)*term2
    return RHS, force, pressure
    

# Function used to determine dudt
#------------------------------------------------------------------------------
def dedt(N,i,index,m,P,rho,v,h,r):
    RHS = 0
    for j in index:
        if j != i:
            RHS += (m * (4*pi)**(-1)) * (P[i]/rho[i]**2) * ( v[j]*dWdrp(r[j],r[i],(h[i]+h[j])/2) + v[i]*dWdr(r[j],r[i],(h[i]+h[j])/2) )
    return RHS


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
N = 5000
m = (M_vir/N)


# Sample initial density distribution, initialize r_i/v_i/P_i
#------------------------------------------------------------------------------
M_samples = np.linspace(M_vir/N,M_vir,N)
steps = 500
r = np.zeros((steps+1,N))
v = np.zeros((steps+1,N))
P = np.zeros((steps+1,N))
for j in range(0,N):
    r[0,j] = fsolve(r_func, 1, args=(M_samples[j],r_s,rho_s))[0]
    P[0,j] = P_NFW(r[0,j],r_s,rho_s)


# Initialize smoothing lengths h_i
#------------------------------------------------------------------------------
eta_new = 1.5
h = np.zeros((steps+1,N))
for j in range(N):
    h[0,j] = eta_new * m * (rho_NFW(r[0,j],r_s,rho_s) * r[0,j]**2)**(-1)
#    h[0,j] = 0.5

#n = 1000
#r_j = r[0,10]
#r_i = np.linspace(r[0,0],r[0,25],n)
#kernel = [W_3S1(r_j,x,h[0,0]) for x in r_i]
#kernel_deriv = np.zeros(n)
#for i in range(0,n-1):
#    kernel_deriv[i] = (kernel[i+1] - kernel[i])/(r_i[i+1] - r_i[i]) 
#
#inflect = 0
#slope = (kernel[1] - kernel[0])/(r_i[1] - r_i[0])    
#for i in range(1,n-1):
#    slope_2 = (kernel[i+1] - kernel[i])/(r_i[i+1] - r_i[i]) 
#    if slope_2 < slope:
#        inflect = r_i[i]
#        break
#    else:
#        slope = (kernel[i+1] - kernel[i])/(r_i[i+1] - r_i[i])
#        
#inflect2 = 0
#slope = (kernel[1] - kernel[0])/(r_i[1] - r_i[0])    
#for i in range(500,n-1):
#    slope_2 = (kernel[i+1] - kernel[i])/(r_i[i+1] - r_i[i]) 
#    if slope_2 > slope:
#        inflect2 = r_i[i]
#        break
#    else:
#        slope = (kernel[i+1] - kernel[i])/(r_i[i+1] - r_i[i])
#
#zero = 0
#slope = (kernel[1] - kernel[0])/(r_i[1] - r_i[0])    
#for i in range(1,n-1):
#    slope_2 = (kernel[i+1] - kernel[i])/(r_i[i+1] - r_i[i]) 
#    if slope_2<0 and slope>0:
#        zero = r_i[i]
#        break
#    else:
#        slope = (kernel[i+1] - kernel[i])/(r_i[i+1] - r_i[i])
#        
#kernel_deriv_Omang = np.array([dWdr(r_j,x,h[0,0]) for x in r_i])
#plt.clf()
#plt.plot(r_i,kernel,'.',label='W_3S1')
#plt.plot(r_i,kernel_deriv,'.',label='dW/dr finite diff')
#plt.plot(r_i,kernel_deriv_Omang*(0.98),'.',label='98% dW/dr Omang')
#plt.axvline(zero,color='grey',linestyle='--',label='zero')
#plt.axvline(inflect,color='grey',linestyle='--',label='inflect 1')
#plt.axvline(inflect2,color='grey',linestyle='--',label='inflect 2')
#plt.axhline(0,color='k',linestyle='--')
#plt.xscale('log')
#plt.title('W_3S1 and dW/dr for r_j='+str("{:.3f}".format(r_j))+' and h='+str(h[0,0]))
#plt.xlabel('r')
#plt.legend()
#plt.show()


# Calculate/plot SPH density profile, compare it to NFW, initialize u
#------------------------------------------------------------------------------
rho = np.zeros((steps+1,N))
neighbors = np.zeros((steps,N))
for j in range (0,N):
    index = np.where(abs(r[0]-r[0,j]) <= 2*h[0,j])[0]       
    neighbors[0,j] = len(index)
#    for k in index:
#        rho[0,j] += (m * (4*pi)**(-1)) * W_3S1(r[0,k],r[0,j],h[0,j])
for j in range (0,N):
    rho[0,j] = rho_NFW(r[0,j],r_s,rho_s)

print(min(neighbors[0]),max(neighbors[0]),np.mean(neighbors[0]))

e = np.zeros((steps+1,N))
e[0] = (3/2)*(P[0]/rho[0])

epsilon = 1e-10
dp_NFW = np.array([( P_NFW(x+epsilon,r_s,rho_s) - P_NFW(x,r_s,rho_s) ) * (epsilon)**(-1) for x in r[0]])
dp_SPH = np.zeros(N)
for j in range(N):
    index = np.where(abs(r[0]-r[0,j]) <= 2*h[0,j])[0] 
    dp_SPH[j] = dvdt(N,j,index,m,P[0],rho[0],r[0],h[0])[2]

plt.clf()
plt.loglog(r[0],abs(dp_SPH),'.',label='SPH abs (h ='+str(h[0,0])+')')
plt.loglog(r[0],abs(dp_NFW),color='k',label='NFW abs')
plt.title('dP/dr for NFW vs. SPH (N=5000)')
plt.xlabel('r')
plt.legend()
plt.show()

#forces = np.zeros(N)
#pressures = np.zeros(N)
#for j in range(0,N):
#    index = np.where(abs(r[0]-r[0,j]) <= 2*h[0,j])[0]
#    a, forces[j], pressures[j] = dvdt(N,j,index,m,P[0],rho[0],r[0],h[0])
#pressures[4990:N] = -0.875*forces [4990:N]
  
#F_NFW = np.array([-G*M_NFW(x,r_s,rho_s)/x**2 for x in r[0]])
#plt.clf()
#plt.loglog(r[0],abs(p_NFW),linewidth=2,label='NFW pressure abs')
#plt.loglog(r[0],abs(pressures),'.',label='SPH pressures')
#plt.loglog(r[0],abs(F_NFW),label='NFW force abs')
#plt.plot(r[0],forces,'.',label='SPH forces (N ='+str(N)+')')
#plt.plot(r[0],forces+pressures,'.',label='Sum of terms')
#plt.axhline(0,color='k',linestyle='--')
#plt.xscale('log')
#plt.xlabel('r')
#plt.title('Force/pressure terms for NFW vs. SPH (h=0.05)')
#plt.legend()
#plt.show()


## Begin time-evolution
##------------------------------------------------------------------------------
dt = 0.0001
t_elapsed = 0
forces = np.zeros((steps,N))
pressures = np.zeros((steps,N))
p_index = 0
print('step =',0,'  r_0 =', "{:.4f}".format(r[0,p_index]),'  rho_0 =',"{:e}".format(rho[0,p_index]),'  h_0 =',"{:.3f}".format(h[0,p_index]),sep = '\t')
for i in range (0,steps):
    for j in range(0,N):
#         Update r, v, and u
        if j <= N-20:
            index = np.where(abs(r[i]-r[i,j]) <= 2*h[i,j])[0]
            neighbors[i,j] = len(index)
            a, forces[i,j], pressures[i,j] = dvdt(N,j,index,m,P[i],rho[i],r[i],h[i])
            v[i+1,j] = v[i,j] + a*dt
            r[i+1,j] = r[i,j] + v[i,j]*dt
            if r[i+1,j] < 0:
                print(i,j)
                r[i+1,j] = np.abs(r[i+1,j])
            e[i+1,j] = e[i,j] + dedt(N,j,index,m,P[i],rho[i],v[i],h[i],r[i])*dt
        else:
            v[i+1,j] = v[i,j]
            r[i+1,j] = r[i,j]
            e[i+1,j] = e[i,j]
    for j in range(0,N):
#         Update rho, P, and h
        if j <= N-20:
            index = np.where(abs(r[i]-r[i,j]) <= 2*h[i,j])[0]
            rho[i+1,j] = get_rho(N,j,index,r[i+1],h[i],m)
            P[i+1,j] = (2/3) * rho[i+1,j] * e[i+1,j]
        else:
            rho[i+1,j] = rho[i,j]
            P[i+1,j] = P[i,j]   
        h[i+1,j] = h[i,j]
    t_elapsed += dt
    print('step =',i+1,'  r_0 =', "{:.4f}".format(r[i+1,p_index]),'  rho_0 =',"{:e}".format(rho[i+1,p_index]),'  h_0 =',"{:.3f}".format(h[i+1,p_index]),sep = '\t')



plt.clf()
for j in range(0,10):
    plt.plot(r[0:steps+1,j],linestyle='--', marker='o',markersize=3,label='Particle '+str(j))
#    plt.plot(forces[0:steps+1,j],linestyle='--', marker='o',markersize=3,label='Particle '+str(j)+' force')
#    plt.plot(pressures[0:steps+1,j],linestyle='--', marker='o',markersize=3,label='Particle '+str(j)+' pressure')
#r_vals = np.linspace(min(r[-1]),max(r[-1]),1000)
#plt.loglog(r_vals,[rho_NFW(x,r_s,rho_s) for x in r_vals],label='NFW')
#plt.loglog(r[0],rho[0],'.',label='SPH i')
#plt.loglog(r[-1],rho[-1],'.',label='SPH f')
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
plt.show()