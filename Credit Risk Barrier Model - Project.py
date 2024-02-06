# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 19:53:11 2024

@author: Aleksander  Lorenc
"""

import numpy as np
import matplotlib.pyplot as plt

# Aleksander Lorenc, AGH, Faculty of Applied Mathematics, Financial Mathematics, January 2024, Credit Risk

# A company needs to get 500 mln PLN of financing. 
# Company may get w_D\in\{0\%, 20\%, 40\%, 60\% \} of the entire needed money by issuing zero-coupon bonds,
# one year lasting, with yield=k_D=13\% (continuous capitalisation) in PLN.
# The rest of the money could be borrowed in EUR or USD. 
# Today there are currency prices: EUR/PLN=4.34 and USD/PLN=4.01
# Yield for zero-coupon bonds equals k_D1=0.05 and k_D2=0.07 for bonds in EUR and USD, respectively.
# The value of the company is given as: V(t)=V(0)*exp([mu_V-0.5*sigma^2_V]*t+sigma_V*W_V(t)), for t\in[0,T].
# mu_V=0.17;  sigma_V=0.35;   V(0)=V0=500 mln PLN;   
# The currency price is given as:  S_i(t)=S_i(0)*exp([mu_i-0.5*sigma^2_i]*t+sigma_i*W_i(t)), for t\in[0,T].
# i=1 or i=2;   mu_1=0.03;  sigma_1=0.09; mu_2=0.05;  sigma_2=0.13;  Cor(V,EUR)=0.7;    Cor(V,USD)=-0.5
# There is a berrier build-in in the bonds. It is given as:   B_i(t)=S_i(t)*D(0)*exp(k_D2*t-gamma*[T-t]).
# i=1 or i=2;  gamma=0.1;    V(t)=D(t)+E(t)   for all t\in[0,T] 
# If the asset hits the barrier, the bond holders get it, 
# sell 80% of it and lodge it on interest rate r=4% until t=T. 
# THe aim of this project is to choose which weight w_D and  which currency (PLN, USD or EUR) 
# give the best expectation of the return rate \mu_E=[EXPECTATION(E(T))-E(0)]/E(0)
# Use Monte-Carlo on time grids:  t=0,1, ... , 52 (weeks)  or t=0,1, ... , 365 (days)





### INPUT DATA

N=365           # number of time steps we consider
T=1             # time horizon for entire project is one year
NSym=10000      # number of symulations
delta_t=T/N     # length of a single time step

r=0.04          # risk free rate

V0=500          # company value in t=0 (mln PLN)
mu_V=0.17       # V mean
sig_V=0.35      # V standard deviation

S1_0=4.34       # EUR/PLN price
S2_0=4.01       # USD/PLN price
mu_1=0.03       # S1 mean
mu_2=0.05       # S2 mean
sig_1=0.09      # S1 standard deviation
sig_2=0.13      # S2 standard deviation

cor_V_to_EUR=0.7    # correlation between two stochastic processes: V and EUR
cor_V_to_USD=-0.5   # correlation between two stochastic processes: V and USD

k_D=0.13        # yield for bonds in PLN      # rentownoć obligacji w PLN
k_D1=0.05       # yield for bonds in EUR
k_D2=0.07       # yield for bonds in USD

Gamma=0.1
F=1000          # bond price (in every currency: 1000 units) 
                # czyli 1000PLN lub 1000EUR lub 1000USD zależnie od kontekstu
                
            


### TIME AXIS

time_steps=[]
time=[]
for i in range (N+1):
    time.append(i*delta_t)
    time_steps.append(i)


### MONTE-CARLO SIMULATIONS 

# w_D=0 (no debt so currency does not matter)
def NoDebt(V0_given, mu_V_given, sig_V_given, T_given, NSteps_given, NSym_given):
    delta_t=T_given/NSteps_given
    V=np.zeros(NSteps_given+1)
    B=np.zeros(NSteps_given+1) 
    V[0]=V0_given
    E_sum=0
    for i in range (0,NSym_given):
        for t in range (0,NSteps_given):
            V[t+1]=V[t]*np.exp((mu_V_given-0.5*sig_V_given*sig_V_given)*delta_t+sig_V_given*np.sqrt(delta_t)*(np.random.normal()))
            if (V[t+1]<=0):         # Barrier=0=const, cause D0=0
                hit=t+1             # \tau=stopping moment, maybe we'll use it
                E_value=0.2*V[t+1]  # Comp. Equity value
                E_sum=E_sum+E_value
                break               # exit the for loop, cause the barrier was hit
        E_value=V[NSteps_given]
        E_sum=E_sum+E_value
    E_exp=E_sum/NSym_given
    Return_rate=E_exp/V0_given-1
    return(Return_rate, V, B)

x=NoDebt(V0, mu_V, sig_V, T, N, NSym)
print("Average Return Rate For NoDebt Scenario = ",100*x[0],"%")  # result around 18.8%

plt.plot(time_steps,x[1])
plt.plot(time_steps,x[2]) 
plt.xlabel("Days") 
plt.ylabel("V and B")
plt.legend(["V","B"])
plt.title("No Debt Scenario")
plt.show()   
      
    
    
    
# w_D!=0, PLN
def PLNDebt(V0_given, mu_V_given, sig_V_given, T_given, NSteps_given, NSym_given, w_D_given, F_given, Gamma_given, k_D_given, r_given):
    delta_t=T_given/NSteps_given
    V=np.zeros(NSteps_given+1)
    B=np.zeros(NSteps_given+1)
    #DEBT=np.zeros(NSteps_given+1)
    V[0]=V0_given
    E_sum=0
    D0=w_D_given*V0_given
    NumOfBonds=D0/(np.exp(-k_D_given*T_given)*F_given)
    B[0]=D0*np.exp(-Gamma_given*T_given)
    #NofHits=0
    #DEBT[0]=D0
    for i in range (0,NSym_given):
        for t in range (0,NSteps_given):
            hit=NSteps_given+1000 # anything greater than NSteps
            V[t+1]=V[t]*np.exp((mu_V_given-0.5*sig_V_given*sig_V_given)*delta_t+sig_V_given*np.sqrt(delta_t)*(np.random.normal()))
            B[t+1]=D0*np.exp(k_D_given*t*delta_t-Gamma_given*(T_given-t*delta_t))
            #DEBT[t+1]=w_D_given*V[t+1]
            if (V[t+1]<=B[t+1]):     # it is the lower barrier so V should be above
                hit=t+1              # hit=\tau=stopping moment
                break
        if (hit<=NSteps_given+1):    # if wehit barrier before T, we part money
            #NofHits=NofHits+1
            deposit=0.8*V[hit]*np.exp(r_given*(T_given-hit*delta_t))
            E_value=0.2*V[hit]
            if(deposit>F_given*NumOfBonds): # we pay the debt
                E_value=E_value+deposit-F_given*NumOfBonds
            else:
                if(E_value>F_given*NumOfBonds-deposit):
                    E_value=E_value+(deposit-F_given*NumOfBonds)
                else:
                    E_value=0
            E_sum=E_sum+E_value
        else:
            D=D0*np.exp(k_D_given*T_given)
            E_sum=E_sum+V[NSteps_given]-D
    E_exp=E_sum/NSym_given
    Return_rate=E_exp/((1-w_D_given)*V0_given)-1
    return(Return_rate,V,B)#,DEBT)
 
               

x=PLNDebt(V0, mu_V, sig_V, T, N, NSym, 0.6, F, Gamma, k_D, r)
print("Average Return Rate For PLNDebt Scenario = ",100*x[0],"%")

# result for w_D=0.2 is around 19.5% , for w_D=0.4 is around 22%, for w_D=0.6 is around  26%

plt.plot(time_steps,x[1])
plt.plot(time_steps,x[2]) 
plt.xlabel("Days") 
plt.ylabel("V and B in PLN")
plt.legend(["V","B"])
plt.title("PLN Debt Scenario")
plt.show()              
#plt.savefig("output.png")            
    
    
    
    
# w_D!=0, EUR or USD
def ForeignDebt(V0_given, mu_V_given, sig_V_given, T_given, NSteps_given, NSym_given, w_D_given, F_given, Gamma_given, k_D_S_given, r_given, S0_given, mu_S_given, sig_S_given, cor_given):
    delta_t=T_given/NSteps_given
    V=np.zeros(NSteps_given+1)
    B=np.zeros(NSteps_given+1)
    S=np.zeros(NSteps_given+1)
    #DEBT=np.zeros(NSteps_given+1)
    V[0]=V0_given#/S0_given
    E_sum=0
    D0=w_D_given*V[0]/S0_given
    NumOfBonds=D0/(np.exp(-k_D_S_given*T_given)*F_given)
    B[0]=(D0*np.exp(-Gamma_given*T_given))*S0_given
    S[0]=S0_given
    #NofHits=0
    #DEBT[0]=D0
    for i in range (0,NSym_given):
        for t in range (0,NSteps_given):
            hit=NSteps_given+1000 # anything greater than NSteps
            dW_PLN=np.random.normal()
            dW_For=np.random.normal()
            dW_Cor=cor_given*dW_PLN+np.sqrt(1-cor_given*cor_given)*dW_For
            V[t+1]=V[t]*np.exp((mu_V_given-0.5*sig_V_given*sig_V_given)*delta_t+sig_V_given*np.sqrt(delta_t)*dW_PLN)
            S[t+1]=S[t]*np.exp((mu_S_given-0.5*sig_S_given*sig_S_given)*delta_t+sig_S_given*np.sqrt(delta_t)*dW_Cor)
            Bar=D0*np.exp(k_D_S_given*t*delta_t-Gamma_given*(T_given-t*delta_t))
            B[t+1]=Bar*S[t+1]#/S[t]
            #DEBT[t+1]=w_D_given*V[t+1]
            if (V[t+1]<=B[t+1]):     # it is the lower barrier so V should be above
                hit=t+1              # hit=\tau=stopping moment
        if (hit<=NSteps_given):    # if wehit barrier before T, we part money
            #NofHits=NofHits+1
            deposit_PLN=0.8*V[hit]*np.exp(r_given*(T_given-hit*delta_t))#*S[hit]
            deposit_For=deposit_PLN/S[NSteps_given]
            E_value=0.2*V[hit]/S[NSteps_given] #################################
            if(deposit_For>F_given*NumOfBonds): # we pay the debt
                E_value=E_value+deposit_For-F_given*NumOfBonds
            else:
                if(E_value>F_given*NumOfBonds-deposit_For):
                    E_value=E_value+(deposit_For-F_given*NumOfBonds)
                else:
                    E_value=0
            E_sum=E_sum+E_value*S[NSteps_given]
        else:
            D=D0*np.exp(k_D_S_given*T_given)
            E_sum=E_sum+(V[NSteps_given]/S[NSteps_given]-D)*S[NSteps_given]######
    E_exp=E_sum/NSym_given
    Return_rate=E_exp/((1-w_D_given)*V0_given)-1
    return(Return_rate,V,B,S)#DEBT,S)


# EUR
x=ForeignDebt(V0, mu_V, sig_V, T, N, NSym, 0.6, F, Gamma, k_D1, r, S1_0, mu_1, sig_1, cor_V_to_EUR)
print("Average Return Rate For ForDebt Scenario = ",100*x[0],"%")

# result for w_D=0.2 is around  20.9%, for w_D=0.4 is around 25.7%, for w_D=0.6 is around  33.4% 

plt.plot(time_steps,x[1])
plt.plot(time_steps,x[2]) 
plt.xlabel("Days") 
plt.ylabel("V and B in EUR")
plt.legend(["V","B"])
plt.title("EUR Debt Scenario")
plt.show() 



# USD
y=ForeignDebt(V0, mu_V, sig_V, T, N, NSym, 0.6, F, Gamma, k_D2, r, S2_0, mu_2, sig_2, cor_V_to_USD)
print("Average Return Rate For ForDebt Scenario = ",100*y[0],"%") 

# result for w_D=0.2 is around  20%, for w_D=0.4 is around 23%, for w_D=0.6 is around  31% 

plt.plot(time_steps,y[1])
plt.plot(time_steps,y[2])
plt.xlabel("Days") 
plt.ylabel("V and B in USD")
plt.legend(["V","B"])
plt.title("USD Debt Scenario")
plt.show() 


# !
# CONCLUSION:  the best relust we get for w_D=0.6 in EUR, in general the higher w_D the better result
# !





#################################################################
#################################################################
# INITIAL VERSION OF SINGLE SYMULATIONS OF TRAJECTORIES

def Vsym(V0,mu_V, sig_V, T, N):
    delta_t=T/N     # length of a single time step
    V=np.zeros(N+1)
    V[0]=V0#*np.exp((mu_V-0.5*sig_V*sig_V)*delta_t+sig_V*np.random.normal())
    for t in range(N):
        V[t+1]=V[t]*np.exp((mu_V-0.5*sig_V*sig_V)*delta_t+sig_V*np.sqrt(delta_t)*(np.random.normal()))
    return(V)

def Ssym(S_0, mu, sig, T, N):
    delta_t=T/N     # length of a single time step
    S=np.zeros(N+1)
    S[0]=S_0#*np.exp((mu-0.5*sig*sig)*delta_t+sig*np.random.normal())
    for t in range(N):
        S[t+1]=S[t]*np.exp((mu-0.5*sig*sig)*delta_t+sig*np.sqrt(delta_t)*np.random.normal())
    return(S)
 

V=Vsym(V0,mu_V, sig_V, T, N)
plt.plot(time_steps,V) 
plt.xlabel("Time steps") 
plt.ylabel("Symulated V")
plt.show()       

#################################################################
#################################################################
  
    