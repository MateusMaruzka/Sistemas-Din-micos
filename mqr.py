# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 18:22:36 2020

@author: maruzka
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import math



"""

Continuous FOPDT

T * dy(t)/dt + y(t) = u(t-O)

Z-transform

{T*z^-1 + 1 = z^-O}Y(z) = Kp*U(z)*z^-O 



Modelagem via minimos quadrados


Sistemas linear do tipo:
    
    A(z)*Y(Z) = z^d * B(z)*U(z)
    
    A(z) = 1 + a0*z^-1 + ... + a_na*z^-na -> Polos 
    B(z) = b0 + b0*z^-1 + ... + b_nb*z^-nb -> Zeros


    y(t) = - a0*y(t-1) -a1*y(t-2) - ... -a_na*y(t-na)
           + b0*u(t-d) + b1*u(t-d-1) + ... + b_nb*u(t-d-nb)
           
           
    Phi(t-1)^T = [-y(t-1) -y(t-2) ... -y(t-na) u(t-d) u(t-d-1) ... u(t-d-nb)]
    
    Theta(t) = [a1 a2 a3 ... a_na b0 b1 b2 b3 ... b_nb]
    
    y(t) = Phi(t-1)^T @ Theta(t) -> Y estimado
    
    J(Phi) = sum L^(N-t)[(y(t) - Phi(t-1)^T @ Theta(t))]
    
    
    Theta(t) = Theta(t-1) + K(t)*{y(t) - y_estimado(t)}
    
    K(t) = P(t-1) * Phi(t-1) / L + Phi^T * P * Phi
    
    P(t) = (1/L) * [I - K Phi P ]
    

"""


def main():
    
    
    # Variaveis de simulação
    
    tf = 20
    ts = 0.1
    
    # Sistema a ser identificado
    
    G = scipy.signal.TransferFunction([1], [1, 4 ,6 ,4 ,1])
    Gd = G.to_discrete(ts, method="zoh")
    print(Gd)
    
    Bp = Gd.num                  # zeros
    Ap = Gd.den                  # poles
    Pnb = len(Bp) - 1             # number of zeros
    Pna = len(Ap) - 1             # number of poles
        
    d = 0

    slack = np.amax([Pna, Pnb]) + 1 + d # slack for negative time indexing of arrays
    kend = math.ceil(tf/ts) + 1   # end of simulation in discrete time
    kmax = kend + slack           # total simulation array size
    
    # Modelo do sistema 
    
    na = 2
    nb = 2
    
    A = np.ones(na)
    B = np.ones(nb)
    
    # jeito de identificar 
    Theta = np.ones(nb+na)/10
    P = np.identity(na+nb)*100
    L = 2
    
    # coisas 
    
    u = 50*np.ones(kmax) + 0*np.random.randn(kmax)
    y = np.zeros(kmax)
    y_est = np.zeros(kmax)
    e = np.zeros(kmax)
    r = np.ones(kmax)


    fig, ax = plt.subplots(ncols=1, nrows=3, constrained_layout=True, figsize = (9,6))

    # Simulate
    for k in range(slack, kmax):
        
        
        # y[k] = np.dot(Bp, u[k-1-d:k-1-d-(Pnb+1):-1]) - np.dot(Ap[1:], y[k-1:k-1-Pna:-1])
        y[k] = -0.5*y[k-1] - 0.05*y[k-2] + 0.3*u[k-1-d] + 0.3*u[k-2-d]
    
        # Phi = np.array([-y[k - i] for i in range(na)] + [u[k - d - j] for j in range(nb)])
        Phi = np.array([-y[k-1], -y[k-2], u[k-1-d], u[k-2-d]])
 
        y_est[k] = np.dot(Phi, Theta)

        e[k] = y[k] - y_est[k]
        
        K_matrix =  np.dot(P, Phi.T)  / np.dot(L + np.dot(Phi.T, P), Phi)
     
        Theta = Theta + K_matrix.T * e[k]
        P = P - K_matrix*(1 + Phi * P * Phi.T)*K_matrix.T
        # P = P - np.dot(np.dot(K_matrix, 1 + np.dot(np.dot(Phi, P), Phi.T)), K_matrix.T)


    
    y = y[slack:]
    u = u[slack:]
  
    print(Theta)
    ax[0].plot(y)
    ax[0].plot(e)
    ax[1].plot(u)
    autocorr_e = np.correlate(e, e, mode='full')
    ax[2].stem(autocorr_e[autocorr_e.size//2:2*autocorr_e.size//3]/np.sum(e**2), use_line_collection=True)
    # ax[2].acorr(e)


if __name__ == "__main__":
    main()
