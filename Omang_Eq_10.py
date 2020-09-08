# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 09:17:55 2020

@author: nebue
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import math


def C(q):
    return q**2 - (3/4)*q**4 + (3/10)*q**5

def D(q):
    return 2*q**2 - 2*q**3 + (3/4)*q**4 - (1/10)*q**5

def W1_integrand(rp,r,h):
    σ = r/h
    σ_prime = rp/h
    X = σ_prime + σ
    Y = abs(σ_prime - σ)
    prefactor = 1/(h*rp*r)
    return prefactor * ( C(X) - C(Y) ) * rp**2

def W2_integrand(rp,r,h):
    σ = r/h
    σ_prime = rp/h
    X = σ_prime + σ
    Y = abs(σ_prime - σ)
    prefactor = 1/(h*rp*r)

    if 0 <= Y < 1:
        return prefactor * ( (-1/10) + D(X) - C(Y) ) * rp**2
    elif 1 <= Y < 2:
        return prefactor * ( D(X) - D(Y) ) * rp**2
    else:
        return 0
    
def W3_integrand(rp,r,h):
    σ = r/h
    σ_prime = rp/h
    Y = abs(σ_prime - σ)
    prefactor = 1/(h*rp*r)
    
    if 0 <= Y < 1:
        return prefactor * ( (7/10) - C(Y) ) * rp**2
    elif 1 <= Y < 2:
        return prefactor * ( (8/10) - D(Y) ) * rp**2
    else:
        return 0


plt.clf()
epsilon = 1e-12
truncate_factor = 5
h = 1
r = (5/4) * h
X = h-r
Y = 2*h-r

if X > 0:
    print('Case 1')
    rp1 = np.linspace(epsilon,X,500)
    rp2 = np.linspace(X+epsilon,Y,500)
    rp3 = np.linspace(Y+epsilon,truncate_factor*h,500)
    W1 = [W1_integrand(x,r,h) for x in rp1]
    W2 = [W2_integrand(x,r,h) for x in rp2]
    W3 = [W3_integrand(x,r,h) for x in rp3]
    plt.plot(rp1,W1,label='W1')
    plt.plot(rp2,W2,label='W2')
    plt.plot(rp3,W3,label='W3')
    integral_1 = quad(W1_integrand, epsilon, X, args=(r,h))[0]
    integral_2 = quad(W2_integrand, X+epsilon, Y, args=(r,h))[0]
    integral_3 = quad(W3_integrand, Y+epsilon, truncate_factor*h, args=(r,h))[0]
    integral_tot = integral_1 + integral_2 + integral_3
    print(integral_tot)
    
elif Y > 0:
    print('Case 2')
    rp2 = np.linspace(epsilon,Y,500)
    rp3 = np.linspace(Y+epsilon,truncate_factor*h,500)
    W2 = [W2_integrand(x,r,h) for x in rp2]
    W3 = [W3_integrand(x,r,h) for x in rp3]
    plt.plot(rp2,W2,label='W2')
    plt.plot(rp3,W3,label='W3')
    integral_2 = quad(W2_integrand, epsilon, Y, args=(r,h))[0]
    integral_3 = quad(W3_integrand, Y+epsilon, truncate_factor*h, args=(r,h))[0]
    integral_tot = integral_2 + integral_3
    print(integral_tot)
    
elif Y <= 0:
    print('Case 3')
    rp3 = np.linspace(epsilon,truncate_factor*h,500)
    W3 = [W3_integrand(x,r,h) for x in rp3]
    plt.plot(rp3,W3,label='W3')
    integral_3 = quad(W3_integrand, epsilon, factor*h, args=(r,h))[0]
    integral_tot = integral_3
    print(integral_tot)
    
plt.legend()
plt.show()