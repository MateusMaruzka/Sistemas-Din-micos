# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 00:16:21 2020

@author: maruzka
"""

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import math



def main():
    
    # Tempo de simulação
    Tf = 400
    Ts = 0.6
    
    # Processo real
    Gp = signal.TransferFunction([1],np.convolve([1,2,1], [1,2,1]))
    # Gp= signal.TransferFunction([1],[1, 5])
    Gpd = Gp.to_discrete(Ts)
    
    # Polinomios A e B do processo real. Polos e zeros
    A = Gpd.den
    B = Gpd.num
    
    # numero de polos e zeros
    Gna = A.size - 1 
    Gnb = B.size - 1
    
    
    #Comportamento desejado -> artigo do silveira
    Bm = np.array([0.1])
    Am = np.array([1, -0.90])
    
    pnb = len(Bm) - 1
    pna = len(Am) - 1
    
    d = 2
    
    
    # Parâmetros para simulação
    slack = np.amax([Gna, Gnb]) + 1 + d
    kend = math.ceil(Tf/Ts) + 1   
    kmax = kend + slack           
    
    
    y = np.zeros(kmax)
    
    u = np.zeros(kmax)  
    
    e = np.zeros(kmax)
    
    r = np.ones(kmax)
    r[kmax//3:] = 2 
    r[kmax*2//3:] = 1
    
    
    # Mínimos quadrados recursivo 

    e_mqr = np.zeros(kmax)
    y_mqr = np.zeros(kmax)
    u_mqr = np.zeros(kmax)
    
    y_abm = np.zeros(kmax)
    u_bm = np.zeros(kmax)

    
    # Modelo da resposta desejada

    # pmf = signal.TransferFunctionDiscrete([0.01, 0],[1 ,-0.99])
    

    Theta = np.ones(3)/10
    P = np.identity(3)*10
    L = 5
    
    kp_t = np.zeros(kmax)
    ki_t = np.zeros(kmax)
    kd_t = np.zeros(kmax)
    
    for k in range(slack, kmax):
        
        
       # Saída do processo real
       y[k] = -np.dot(A[1:], y[k-1:k-1-Gna:-1]) + np.dot(B, u[k-1:k-1-Gnb-1:-1]) # Sem utilizar atraso D  

       e[k] = r[k] - y[k] # Erro. Referencia - Saída do processo
       
       
       # Estimador

       # y_abm(t) = [Am - z^-d * Bm] * y(t)
       # A(z) = 1 + a0*z^-1 + ... + a_na*z^-na -> Polos 
       # B(z) = b0 + b1*z^-1 + ... + b_nb*z^-nb -> Zeros
       
       # P = P - K_matrix*(1 + Phi * P * Phi.T)*K_matrix.T

    
       y_abm[k] = np.dot(Am, y[k:k-pna-1:-1]) - np.dot(Bm, y[k-d:k-d-pnb-1:-1])
       u_bm[k] = np.dot(Bm, u[k-d:k-len(Bm)-d:-1] - u[k-1-d:k-1-len(Bm)-d:-1]) 

       Phi = y_abm[k:k-3:-1]
       u_mqr[k] = np.dot(Phi, Theta)
       e_mqr[k] = u_bm[k] - u_mqr[k]
              
       # Ganho e Matriz de covariancia
       K_matrix = np.dot(P, Phi.T) / (np.dot((np.dot(Phi.T, P)), Phi) + L)
       P = P - np.dot(np.dot(K_matrix, 1 + np.dot(np.dot(Phi, P), Phi.T)), K_matrix.T)
 
       # Atualização dos parâmetros 
       Theta = Theta + K_matrix.T * e_mqr[k]

    
       s0, s1, s2 = Theta
       
       # Ganhos obtidos através do controlador PID discretizado através
       # de backwards difference
       kc = -s1-2*s2
       ki = s0+s1+s2
       kd = s2
       
       # Foward
       # kc = s0-s2
       # ki = s0+s1+s2
       # kd = s2
       
       kp_t[k] = kc
       ki_t[k] = ki
       kd_t[k] = kd
       
       # du = 0.1*(e[k] - e[k-1]) + 0.1*e[k] + (0.1)*(e[k] - 2*e[k-1] + e[k-2])
       # u[k] = u[k-1] + du
       u[k] = u[k-1] + s0*e[k] + s1*e[k-1] + s2*e[k-2]




    # y = y[slack-d:]    
    # u = u[slack-d:]
    # r = r[slack-d:]
    
    
    print(kc,ki,kd)
    fig, ax = plt.subplots(ncols = 1, nrows = 3, sharex=True)
    
    ax[0].plot(y)
    ax[0].plot(r, "k--")
    ax[1].plot(u)
       
    ax[0].set_ylabel("y(t)")
    ax[1].set_ylabel("u(t)")
    
    ax[2].plot(kp_t, label = "Kp")
    ax[2].plot(ki_t, label = "Ki")
    ax[2].plot(kd_t, label = "Kd")
    plt.legend()
    
    
    plt.xlim([0, kend])
    plt.ion()
      
if __name__ == "__main__":
    np.random.seed(0)
    main()