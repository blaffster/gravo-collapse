# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 09:17:55 2020

@author: nebue
"""

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

h = 1
r = (7/2) * h
epsilon = 1e-12

integral_1 = quad(W1_integrand, epsilon, h-r, args=(r,h))[0]
integral_2 = quad(W2_integrand, h-r+epsilon, 2*h-r, args=(r,h))[0]
integral_3 = quad(W3_integrand, 2*h-r+epsilon, math.inf, args=(r,h))[0]
integral_tot = integral_1 + integral_2+ integral_3
print(integral_tot)