# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 16:44:45 2020

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
        

h = 1
multiple = 1/6
rp = multiple*h
if multiple < 1:
    r = np.linspace(1e-12,(1/multiple)*h,500)
else:
    r = np.linspace(1e-12,2*multiple*h,500)
kernel = [W_3S1(rp,x,h) for x in r]
i_max = np.argmax(kernel)
r_max = r[i_max]
print(r_max)
kernel_dr = [dWdr(rp,x,h) for x in r]
plt.clf()
plt.title('h='+str(h)+', rp ='+str(round(rp,3))+'h')
plt.xlabel('r')
plt.axvline(r_max,linestyle='--',color='purple',label='r_max='+str(round(r_max,3)))
if 1-rp > 0:
    plt.axvline(1-rp,linestyle='--',color='red',label='r='+str(round(1-rp,3)))
if 2-rp > 0:
    plt.axvline(2-rp,linestyle='--',color='green',label='r='+str(round(2-rp,3)))
plt.axhline(0,linestyle='--',color='grey')
plt.plot(r,kernel,label='kernel')
plt.plot(r,kernel_dr,label='kernel_dr')
plt.legend()
plt.show()


#h = 1
#multiple = 1/3
#r = multiple * h
#if multiple < 1:
#    rp = np.linspace(1e-12,(1/multiple)*h,500)
#else:
#    rp = np.linspace(1e-12,2*multiple*h,500)
#kernel = [W_3S1(x,r,h) for x in rp]
#i_max = np.argmax(kernel)
#r_max = rp[i_max]
#print(r_max)
#kernel_drp = [dWdrp(x,r,h) for x in rp]
#plt.clf()
#plt.title('h='+str(h)+', r ='+str(round(r,3))+'h')
#plt.xlabel('rp')
#plt.axvline(r_max,linestyle='--',color='purple',label='r_max='+str(round(r_max,3)))
#if 1-r > 0:
#    plt.axvline(1-r,linestyle='--',color='red',label='r='+str(round(1-r,3)))
#if 2-r > 0:
#    plt.axvline(2-r,linestyle='--',color='green',label='r='+str(round(2-r,3)))
#plt.axhline(0,linestyle='--',color='grey')
#plt.plot(rp,kernel,label='kernel')
#plt.plot(rp,kernel_drp,label='kernel_drp')
#plt.legend()
#plt.show()