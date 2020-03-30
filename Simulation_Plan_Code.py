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


def get_Flen(Lce):
    wtemp = np.power(44.2, 2)
    c = 1/wtemp
    eqtemp1 = Lce/0.082
    eqtemp2 = np.power(eqtemp1, 2)
    return np.maximum((c*eqtemp2-2*c*eqtemp1+c+1),0)


def Vfact(q):
    return np.minimum((10/3)*q , 1)


def get_Fcerel(Lsee):
    return force_SEE(0)/Fmax #Need to pass Lsee, WIP


def Vcerel(q, Lm):
    """
    Need: P1, P2, P3, FceRel, q, Vfact,
    """


def get_P1():
    """
    Need: Vfact, Flen, P2,
    """


def get_P2():
    """
    Need: Flen, Fasympt (Const)
    """


def get_P3():
    """
    Need: P1, P2, Flen
    """


def get_Vfact():
    """
    Need: q
    """


def Lm():
    # ankleangle depends on moment, so need moment curve
    ankleangle = 1
    return 0.464+(-0.037*ankleangle)


def length_CE():
    # Alison add regression model here.
    return 0

def length_SEE(Lm,Lce):
    return Lm - Lce


# def get_moment(force):
#     return r*force


ode_handle = lambda t, x: qdot(t, x)

obj = solve_ivp(ode_handle, [0, 2], [1], rtol=1e-5, atol=1e-6)
# [0, 2] is the time that we're integrating, [0] is our initial condition y0.
plt.plot(obj.t, obj.y[0])
plt.title("Active State vs. Time")
plt.ylabel("Active State (q)")
plt.xlabel("Time (t)")
plt.show()
