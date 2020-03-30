import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

Fmax = 1528
Lceopt = 0.082
w = 44.2
Lslack = 0.317
A0 = 0.464
dankle = -0.037
Umax = 0.04
Arel = 0.41
Brel = 5.2
stim_index = 0

# This is temporary, eventually we'll need to design a sophisticated stim profile. 
stim_profile = np.zeros(500)


def qdot(stim, q):
    global stim_index
    ms = 0.001
    t2 = 1/(65*ms)
    t1 = (1/55*ms) - t2

    stim_index += 1
    return (stim_profile[stim_index]-q) * (t1*stim_profile[stim_index]+t2)


def force_SEE(Lsee):
    if Lsee > Lslack:
        num = Lm() - length_CE() - Lslack # LCE comes from regression model (WIP)
        den = Umax * Lslack
        return Fmax * (np.power(num / den, 2))
    else:
        return 0


def force_length_CE(Lce):
    wtemp = np.power(44.2, 2)
    c = 1/wtemp
    eqtemp1 = Lce/0.082
    eqtemp2 = np.power(eqtemp1, 2)
    return np.maximum((c*eqtemp2-2*c*eqtemp1+c+1),0)


def Vfact(q):
    return np.minimum((10/3)*q , 1)


def Fcerel(Lsee,Lslack):
    return force_SEE(0)/1538 #Need to pass Lsee, WIP


# def Vcerel(q,Lm, length_CE):
#     Lce = length_CE()
#     if (Fcerel(Lsee,Lslack)/q)<force_length_CE(Lce):
#         return -Vfact(q)*5.2*((force_length_CE(Lce)+0.41)/((Fcerel(Lsee,Lslack)/q)+0.41-1)
#     else:
#         P2=force_length_CE(Lce)*1.5
#         eqtemp1 = np.power((force_length_CE(Lce)+P2),2)
#         P1=(Vfact()*5.2*eqtemp1)/((force_length_CE(Lce)/0.41)/2)S
#         P3=P1/(force_length_CE(Lce)+P2)
#         eq2temp=np.sqrt(P1/(Vfact()*200),2)
#         if (Fcerel()/q)<(eq2temp-2):
#             return (-P1/(Fcerel()/q)+P2)+P3
#         else:
#             return (Vfact()*200*((Fcerel/q)+P2)+P3+2*np.sqrt(P1*Vfact()*200,2)


def Lm():
    # ankleangle depends on moment, so need moment curve
    ankleangle = 1
    return 0.464+(-0.037*ankleangle)


def length_CE():
    # Alison add regression model here.
    return 0


# def get_moment(force):
#     return r*force


ode_handle = lambda t, x: qdot(t, x)

obj = solve_ivp(ode_handle, [0, 2], [1], rtol=1e-5, atol=1e-6)
# [0, 2] is the time that we're integrating, [0] is our initial condition y0.
plt.plot(obj.t, obj.y[0])
plt.show()
