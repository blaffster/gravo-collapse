# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 13:13:30 2020

@author: nebue
"""

import numpy as np
import matplotlib.pyplot as plt
pi = np.pi



def C(q):
    return q**2 - (3/4)*q**4 + (3/10)*q**5


def D(q):
    return 2*q**2 - 2*q**3 + (3/4)*q**4 - (1/10)*q**5


def W_3S1(rp,r,h):
    σ = r/h
    σ_prime = rp/h
    X = σ_prime + σ
    Y = abs(σ_prime - σ)
    print('X=',X,'Y=',Y)
    prefactor = 1/(h*rp*r)
    
    if 2 < X:
        if 0 <= Y < 1:
            return prefactor * ( (7/10) - C(Y) )
        elif 1 <= Y < 2:
            return prefactor * ( (8/10) - D(Y) )
        else:
            print('zilch')
            return 0
        
    elif 1 < X <= 2:
        if 0 <= Y < 1:
            return prefactor * ( (-1/10) + D(X) - C(Y) )
        elif 1 <= Y < 2:
            return prefactor * ( D(X) - D(Y) )
        else:
            print('zilch')
            return 0
        
    elif X <= 1:
        return prefactor * ( C(X) - C(Y) )


def get_xy(N,h,multiplier):
    r = multiplier * (h/2)
    rp = np.linspace(1e-8,2*h+r,N)
    x_vals = np.zeros(N)
    y_vals = np.zeros(N)
    for i in range(N):
        x_vals [i] = (rp[i]-r)/h
        y_vals [i] = ( r*W_3S1(rp[i],r,h) )/h
    return x_vals, y_vals


x1,y1 = get_xy(100,1,1)
x2,y2 = get_xy(100,1,3)
x3,y3 = get_xy(100,1,5)
x4,y4 = get_xy(100,1,7)
plt.clf()
plt.plot(x1,y1*(1/2),label='r=h/2, y-values halved')
plt.plot(x2,y2,label='r=3h/2')
plt.plot(x3,y3,label='r=5h/2')
plt.plot(x4,y4,label='r=7h/2')
plt.ylim(0,0.8)
plt.xlim(-2,2)
plt.yticks(np.arange(0, 0.85, 0.05))
#plt.axhline(0.7,color='grey',linestyle='--')
plt.xlabel('(rp-r)/h')
plt.ylabel('r*W_3S1/h')
plt.legend()
plt.show()