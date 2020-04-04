import operator
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

from scipy import optimize
from scipy.integrate import solve_ivp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.patches import Rectangle

ankle_data = pd.read_csv('355_data/ankle_angle.csv', delimiter=',')
Lce_data = pd.read_csv('355_data/TA_Lce.csv', delimiter=',')
Lslack = 0.380
Umax = 0.04
Fmax = 1528
width = 0.56
c = -1 / np.power(width, 2)
Lce_opt = 0.075

def get_regression( x, y, degree ):
    x = np.transpose(x)
    y = np.transpose(y)

    # transforming the data to include another axis
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]

    polynomial_features= PolynomialFeatures(degree=degree)
    x_poly = polynomial_features.fit_transform(x)

    model = LinearRegression()
    model.fit(x_poly, y)
    y_poly_pred = model.predict(x_poly)

    rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
    r2 = r2_score(y,y_poly_pred)

    print("RMSE for degree " + str(degree) + " is: ", rmse)

    # plt.title("Ankle Angle over Gait Cycle")
    # plt.xlabel("Gait Cycle (%)")
    # plt.grid(True)
    # plt.ylabel("Ankle Angle (Degrees)")
    # plt.scatter(x, y, s=10)
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)
    x, y_poly_pred = zip(*sorted_zip)
    # plt.plot(x, y_poly_pred, color='m')
    # plt.legend()
    # plt.show()

    return model


def get_prediction(model, val, degree):
    val = np.transpose([val])
    val = val[:, np.newaxis]

    polynomial_features = PolynomialFeatures(degree=degree)
    x_poly = polynomial_features.fit_transform(val)

    prediction = model.predict(x_poly)
    return prediction[0][0]


def get_Lm(theta_ank):
    return 0.464 + (-0.037 * math.radians(theta_ank))


def get_force_SEE(Lm, Lce):
    if Lm - Lce > Lslack:
        num = Lm - Lce - Lslack
        den = Umax * Lslack
        return Fmax * (np.power(num / den, 2))
    else:
        return 0


def get_force_len(Lce):
    temp = Lce / Lce_opt
    return np.maximum((c * np.power(temp, 2) - 2 * c * temp + c + 1), 0)


def qdot(t, q):
    ms = 0.001
    t2 = 1/(65*ms)
    t1 = (1/55*ms) - t2

    stim = get_prediction(stim_control_model, t, 1)
    return (stim-q) * (t1*stim+t2)


def ramp_qdot(t, q):
    ms = 0.001
    t2 = 1/(65*ms)
    t1 = (1/55*ms) - t2

    if t < 90:
        stim = get_prediction(ramp_1, t, 1)
    else:
        stim = get_prediction(ramp_2, t, 1)


    return (stim-q) * (t1*stim+t2)


def ramp_qdot_3(t, q):
    ms = 0.001
    t2 = 1/(65*ms)
    t1 = (1/55*ms) - t2

    if t < 90:
        stim = get_prediction(S3_3_ramp_1, t, 1)
    else:
        stim = get_prediction(S3_3_ramp_2, t, 1)

    return (stim-q) * (t1*stim+t2)


def ramp_qdot_2(t, q):
    ms = 0.001
    t2 = 1/(65*ms)
    t1 = (1/55*ms) - t2

    if t < 90:
        stim = get_prediction(S3_2_ramp_1, t, 1)
    else:
        stim = get_prediction(S3_2_ramp_2, t, 1)

    return (stim-q) * (t1*stim+t2)


def validate_qdot(t, q):
    ms = 0.001
    t2 = 1 / (65 * ms)
    t1 = (1 / 55 * ms) - t2

    if t < 75:
        stim = get_prediction(val_model_1, t, 1)
    elif t < 87:
        stim = get_prediction(val_model_2, t, 1)
    else:
        stim = get_prediction(val_model_3, t, 1)

    return (stim-q) * (t1*stim+t2)


def get_vce_rel(handler, IC):
    ankle_angles = []
    lce_vals = []
    lm_vals = []
    fsee_vals = []
    f_len_vals = []
    P1 = []
    P3 = []
    vce_rel = []
    ramp_vce_rel = []

    for i in gait_per:
        ankle_angles.append(get_prediction(ankle_model, i, 5))
        lce_vals.append(get_prediction(length_CE_model, i, 5))

    for i in ankle_angles:
        lm_vals.append(get_Lm(i))

    for i, j in zip(lm_vals, lce_vals):
        fsee_vals.append(get_force_SEE(i, j))

    for i in lce_vals:
        f_len_vals.append(get_force_len(i))

    obj = solve_ivp(handler, [66, 100], [IC], t_eval=[*gait_per], rtol=1e-5, atol=1e-6)

    fce_rel = [force / Fmax for force in fsee_vals]
    P2 = [force * (-1.5) for force in f_len_vals]
    V_fact = [min((10 / 3) * q, 1) for q in obj.y[0]]

    for v, f, p in zip(V_fact, f_len_vals, P2):
        Arel = 0.41
        Brel = 5.2
        slopfac = 2

        num = v * Brel * np.power(f + p, 2)
        den = (f + Arel) / slopfac

        P1.append(num / den)

    for p1, p2, f in zip(P1, P2, f_len_vals):
        P3.append(p1 / (f + p2))

    for vf, fce, q, flen, p1, p2, p3 in zip(V_fact, fce_rel, obj.y[0], f_len_vals, P1, P2, P3):
        if fce / q < flen:
            num = flen + 0.41
            den = fce / q + 0.41
            vce_rel.append(-vf * 5.2 * ((num / den) - 1))
        else:
            hyperbolic = -np.sqrt(p1 / (vf * 200)) - p2
            if fce / q < hyperbolic:
                den = fce / q + p2
                vce_rel.append(-(p1 / den) + p3)
            else:
                term = fce / q + p2
                vce_rel.append(vf * 200 * term + p3 + 2 * np.sqrt(p1 * vf * 200))

    return vce_rel, obj.t, obj.y[0]
    # return 0, 0, 0

gait_per = np.linspace(66, 100, num=35)

gait_ankle = ankle_data.values[:, 0]
ankle_angle = ankle_data.values[:, 1]

gait_CE = Lce_data.values[:, 0]
val_CE = [val * Lce_opt for val in Lce_data.values[:, 1]]

gait_vce = [66.0968661,67.09401709,68.09116809,69.08831909,70.08547009,71.08262108,72.07977208,73.07692308,74.07407407,75.07122507,76.06837607,77.06552707,78.06267806,79.05982906,80.05698006,81.05413105,82.05128205,83.04843305,84.04558405,85.04273504,86.03988604,87.03703704,88.03418803,89.03133903,90.02849003,91.02564103,92.02279202,93.01994302,94.01709402,95.01424501,96.01139601,97.00854701,98.00569801,99.002849,100]
val_vce = [-28.99081595,-36.43184513,-43.87287431,-48.82943144,-50.05928049,-50.04689353,-47.55003451,-43.81093946,-41.31408044,-36.33274938,-43.77377856,-47.48809966,-48.71794872,-44.97885367,-41.23975863,-44.95407974,-42.45722072,-37.47588965,-28.76785051,-5.152979066,19.7041284,39.59229177,68.17610731,70.67296633,58.26299305,43.36854772,39.65422661,24.75978128,8.623099927,8.635486896,7.40563784,7.418024809,-0.023004371,-7.464033551,-12.42059068]

validate_1 = [[66, 75], [0.5, 0.75]]
validate_2 = [[75, 87], [0.75, 0.3]]
validate_3 = [[87, 100], [0.3, 0.8]]

stim_control_x = [66, 100]
stim_control_y = [0.95, 0.95]

stim_ramp =   [[[66, 90], [90, 100]], [[0.33, 0.33], [0.33, 0.33]]]
stim_ramp_2 = [[[66, 90], [90, 100]], [[0.66, 0.66], [0.66, 0.66]]]
stim_ramp_3 = [[[66, 90], [90, 100]], [[0.95, 0.95], [0.95, 0.95]]]
# stim_ramp_2 = [[[66, 85], [85, 95], [95,100]], [[0.305, 0.305], [0.305, 0.95], [0.95, 0.95]]]
# stim_ramp_3 = [[[66, 80], [80, 90], [90,100]], [[0.305, 0.305], [0.305, 0.95], [0.95, 0.95]]]

val_model_1 = get_regression(validate_1[0], validate_1[1], 1)
val_model_2 = get_regression(validate_2[0], validate_2[1], 1)
val_model_3 = get_regression(validate_3[0], validate_3[1], 1)

stim_control_model = get_regression(stim_control_x, stim_control_y, 1) # Rectangle Wave

ramp_1 = get_regression(stim_ramp[0][0], stim_ramp[1][0], 1)
ramp_2 = get_regression(stim_ramp[0][1], stim_ramp[1][1], 1)

S3_2_ramp_1 = get_regression(stim_ramp_2[0][0], stim_ramp_2[1][0], 1)
S3_2_ramp_2 = get_regression(stim_ramp_2[0][1], stim_ramp_2[1][1], 1)
# S3_2_ramp_3 = get_regression(stim_ramp_2[0][2], stim_ramp_2[1][2], 1)

S3_3_ramp_1 = get_regression(stim_ramp_3[0][0], stim_ramp_3[1][0], 1)
S3_3_ramp_2 = get_regression(stim_ramp_3[0][1], stim_ramp_3[1][1], 1)
# S3_3_ramp_3 = get_regression(stim_ramp_3[0][2], stim_ramp_3[1][2], 1)

ankle_model = get_regression(gait_ankle, ankle_angle, 5)
length_CE_model = get_regression(gait_CE, val_CE, 5)

vce_model = get_regression(gait_vce, val_vce, 25)

ode_handle = lambda t, x: qdot(t, x)
ramp_func = lambda t, x: ramp_qdot_3(t, x)
ramp_func_2 = lambda t, x: ramp_qdot_2(t, x)
ramp_func_3 = lambda t, x: ramp_qdot(t, x)
validate_func = lambda t, x: validate_qdot(t, x)

vce_rel, as_t_1, as_y_1 = get_vce_rel(validate_func, 0.3)
# vce_rel_2, as_t_2, as_y_2 = get_vce_rel(ramp_func_2, 0.66)
# vce_rel_3, as_t_3, as_y_3 = get_vce_rel(ramp_func_3, 0.33)

# rmse_1 = round(np.sqrt(mean_squared_error(val_vce, vce_rel)), 2)
# rmse_2 = round(np.sqrt(mean_squared_error(val_vce, vce_rel_2)), 2)
# rmse_3 = round(np.sqrt(mean_squared_error(val_vce, vce_rel_3)), 2)

plt.title("Velocity of CE due to Validation Stimulus over Gait Cycle")
plt.plot(gait_vce, val_vce, linestyle='--', color='r', label='Healthy $V_{CE}$')
plt.ylabel("Velocity (mm/s)")
plt.xlabel("Gait Cycle (%)")
plt.grid(True)
plt.plot(gait_per, vce_rel, label='Modelled $V_{CE}$')
plt.legend()
# r2_1 = round(r2_score(val_vce, vce_rel), 2)
# r2_2 = round(r2_score(val_vce, vce_rel_2), 2)
# r2_3 = round(r2_score(val_vce, vce_rel_3), 2)
#
# plt.subplot(321)
#
# plt.title("Active State Due to Stimuli over Gait Cycle")
# plt.plot(as_t_3, as_y_3)
# plt.grid(True)
# plt.ylabel("Active State")
#
# plt.subplot(322)
#
# plt.title("Relative Velocity of CE Due to Stimuli over Gait Cycle")
# line1, = plt.plot(gait_vce, val_vce, linestyle='--', color='r')
# plt.ylabel("Velocity (mm/s)")
# plt.grid(True)
# line2, = plt.plot(gait_per, vce_rel_3, color='b')
# error = Rectangle((0, 0), 0, 0, fc="w", fill=False, edgecolor='none', linewidth=0)
# error_legend = plt.legend([error, error], ('RMSE = ' + str(rmse_3), '$R^2$ = ' + str(r2_3)), loc='upper left')
# plt.legend((line1, line2), ('Healthy $V_{CE}$', 'Modelled $V_{CE}$'), loc='lower right')
# plt.gca().add_artist(error_legend)
#
# plt.subplot(323)
#
# plt.plot(as_t_2, as_y_2)
# plt.grid(True)
# plt.ylabel("Active State")
#
# plt.subplot(324)
#
# line1, = plt.plot(gait_vce, val_vce, linestyle='--', color='r')
# plt.ylabel("Velocity (mm/s)")
# plt.grid(True)
# line2, = plt.plot(gait_per, vce_rel_2, color='b')
# error = Rectangle((0, 0), 0, 0, fc="w", fill=False, edgecolor='none', linewidth=0)
# error_legend = plt.legend([error, error], ('RMSE = ' + str(rmse_2), '$R^2$ = ' + str(r2_2)), loc='upper left')
# plt.legend((line1, line2), ('Healthy $V_{CE}$', 'Modelled $V_{CE}$'), loc='lower right')
# plt.gca().add_artist(error_legend)
#
# plt.subplot(325)
#
# plt.plot(as_t_1, as_y_1)
# plt.grid(True)
# plt.ylabel("Active State")
# plt.xlabel("Gait Cycle (%)")
#
# plt.subplot(326)
#
# line1, = plt.plot(gait_vce, val_vce, linestyle='--', color='r')
# plt.ylabel("Velocity (mm/s)")
# plt.xlabel("Gait Cycle (%)")
# plt.grid(True)
# line2, = plt.plot(gait_per, vce_rel, color='b')
# error = Rectangle((0, 0), 0, 0, fc="w", fill=False, edgecolor='none', linewidth=0)
# error_legend = plt.legend([error, error], ('RMSE = ' + str(rmse_1), '$R^2$ = ' + str(r2_1)), loc='upper left')
# plt.legend((line1, line2), ('Healthy $V_{CE}$', 'Modelled $V_{CE}$'), loc='lower right')
# plt.gca().add_artist(error_legend)

plt.show()
#

# plt.title("Stimulation Profile over Gait Cycle")
# plt.ylabel("Stimulation (Normalized)")
# plt.xlabel("Gait Cycle (%)")
# plt.grid(True)
# plt.plot([66,90,100],[0.3,0.3,0.95])
# plt.show()
