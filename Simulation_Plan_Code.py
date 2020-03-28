import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

Fmax=1528
Lceopt=0.082
w=44.2
Lslack=0.317
A0=0.464
dankle=-0.037
Umax=0.04
Arel= 0.41
Brel=5.2

#QDOT EQ'N
def qdot(stim, q):
    ms=np.power(10,-3)
    t2= 1/(65*ms)
    t1= (1/55*ms)-t2
    return (stim-q)*(t1*stim+t2)

#INTERMEDIATE EQ'N
def Fse(Lse,Lslack):
    if Lse>0.317:
        return 1528*(np.power((Lm-Lce-0.317)/(0.04*0.0317)),2)
    else :
        return 0
                     
def Flen(Lce):
    wtemp = np.power(44.2,2)
    c=1/wtemp
    eqtemp1 = (Lce/0.082)   
    eqtemp2 = np.power(eqtemp1,2)
    return np.maximum((c*eqtemp2-2*c*eqtemp1+c+1),0)

def Vfact(q):
    return np.minimum((10/3)*q , 1)
                     
def Fcerel(Lse,Lslack):
    return Fse()/1538

def Vcerel(q,Lm, Lce):
    if (Fcerel(Lse,Lslack)/q)<Flen(Lce):
        return -Vfact(q)*5.2*((Flen(Lce)+0.41)/((Fcerel(Lse,Lslack)/q)+0.41-1)
    else:
        P2=Flen()*1.5
        eqtemp1 = np.power((Flen()+P2),2)
        P1=(Vfact()*5.2*eqtemp1)/((Flen()/0.41)/2)
        P3=P1/(Flen()+P2)
        eq2temp=np.sqrt(P1/(Vfact()*200),2)
        if (Fcerel()/q)<(eq2temp-2):
            return (-P1/(Fcerel()/q)+P2)+P3
        else:
            return (Vfact()*200*((Fcerel/q)+P2)+P3+2*np.sqrt(P1*Vfact()*200,2)

def Lm():
    #ankleangle depends on moment, so need moment curve
    M=#write moment eq
    return 0.464+(-0.037*ankleangle)

