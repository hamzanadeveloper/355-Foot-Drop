import operator

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

def Regression( x, y, degree ):

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
    #test_X = [[0, 0, 0, 0, 0, 0]]
    #print(model.predict(test_X))

    rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
    r2 = r2_score(y,y_poly_pred)


    plt.scatter(x, y, s=10)
    # sort the values of x before line plot
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)
    x, y_poly_pred = zip(*sorted_zip)
    plt.plot(x, y_poly_pred, color='m')
    plt.show()

    return model


def find_yVal(model, x, degree):
    x_array = []
    for i in range(degree+1):
        x_array.append(x)
    print(x_array)
    y = model.predict([x_array])
    print(y)
    return y[0][0]


# Flexion Angles and Moments graph

angles = [-29.9193548387096, -25.3225806451612, -23.1451612903225, -21.4516129032258, -19.2741935483871, -17.3387096774193, -15.6451612903225, -13.9516129032258, -12.5, -10.3225806451612, -8.14516129032258, -5.72580645161291, -3.30645161290323, 0.0806451612903202, 2.74193548387096, 9.51612903225805, 11.9354838709677, 14.5967741935483, 17.9838709677419, 21.6129032258064, 23.5483870967741, 25.9677419354838, 27.4193548387096, 29.8387096774193, 19.6774193548386, 7.09677419354838, 4.91935483870967, -27.258064516129]
moments = [71.4117647058823, 70, 68.1176470588235, 66.235294117647, 63.4117647058823, 61.0588235294117, 58.235294117647, 55.8823529411764, 53.5294117647059, 50.235294117647, 47.4117647058823, 45.0588235294117, 43.6470588235294, 42.235294117647, 41.7647058823529, 40.3529411764706, 39.4117647058823, 38, 36.1176470588235, 33.2941176470588, 31.4117647058823, 29.0588235294117, 27.6470588235294, 25.2941176470588, 34.7058823529411, 40.8235294117647, 41.2941176470588, 70.9411764705882]

angle_moment_model = Regression(angles, moments, 5)
tempYval = find_yVal(angle_moment_model,0,5)

# Lsee and forces graph

Lsee = [0.317829515565093, 0.318036970895324, 0.318278976634274, 0.3184520344337, 0.318659642641846, 0.31890180125871, 0.31904023221046, 0.319178816040124, 0.319317170552916, 0.319455525065708, 0.319594032456415, 0.319697989438402, 0.31980194642039, 0.319905903402377, 0.320079114079718, 0.320183071061705, 0.320321731330327, 0.32059874611174, 0.320772109666996, 0.320980176508885, 0.321153692942055, 0.321327056497311, 0.321534817583371, 0.321708181138627, 0.321812443876443, 0.321951257022979, 0.322090070169516, 0.322263586602686, 0.322402399749222, 0.322541365773673, 0.322714882206843, 0.322923101926647, 0.323096771237732, 0.323235890140097, 0.323443956981987, 0.323617626293072, 0.323722041908803, 0.323826457524535, 0.323931026018181, 0.32413909286007, 0.324243508475802, 0.324347771213619, 0.324486584360155, 0.32462570326252, 0.3247649750428, 0.324904093945166, 0.325078221889995, 0.325252044078994, 0.325426019145909, 0.325599994212823, 0.325739418871018, 0.325878843529213, 0.326052971474042, 0.326226946540956, 0.326435777772419, 0.326610058595162, 0.326784339417906, 0.326923764076101, 0.327028332569747, 0.327202460514576, 0.327307181886137, 0.327411903257698, 0.327550869282148, 0.327690141062429, 0.327829871476453, 0.327934745725928, 0.328074017506209, 0.328247992573123, 0.328353019700513, 0.328492597236623, 0.328597318608184, 0.328702345735574, 0.328841770393769, 0.328946338887415, 0.329016356972342, 0.329121384099732, 0.329260808757927, 0.329400233416122, 0.329505260543512, 0.329644532323792, 0.329749712329097, 0.329819883291938, 0.329959460828048, 0.330098885486243, 0.330169056449084, 0.330273624942731, 0.330343795905572, 0.33037895782595, 0.330414119746328, 0.330483832075426, 0.330553544404523, 0.330623715367365, 0.330693886330206, 0.330833158110486, 0.330903176195413, 0.330938338115791, 0.330973805791999, 0.33107822140773, 0.331148698126401, ]
forces = [3.38053097345141, 6.7610619469026, 10.141592920354, 16.9026548672566, 23.6637168141592, 30.4247787610618, 35.4955752212388, 43.946902654867, 47.3274336283184, 50.70796460177, 57.4690265486726, 64.2300884955752, 70.9911504424778, 77.7522123893802, 87.8938053097344, 94.654867256637, 104.796460176991, 118.318584070796, 131.840707964601, 148.743362831858, 165.646017699114, 179.16814159292, 189.309734513274, 202.831858407079, 216.353982300884, 229.87610619469, 243.398230088495, 260.300884955752, 273.823008849557, 290.725663716814, 307.62831858407, 327.911504424778, 348.194690265486, 368.477876106194, 385.380530973451, 405.663716814159, 422.566371681416, 439.469026548672, 459.75221238938, 476.654867256637, 493.557522123893, 507.079646017699, 520.601769911504, 540.884955752212, 564.548672566371, 584.831858407079, 615.256637168141, 638.920353982301, 665.964601769911, 693.008849557522, 720.053097345132, 747.097345132743, 777.522123893805, 804.566371681415, 838.371681415929, 872.176991150442, 905.982300884955, 933.026548672566, 953.309734513274, 983.734513274336, 1007.39823008849, 1031.06194690265, 1047.96460176991, 1071.62831858407, 1105.43362831858, 1132.47787610619, 1156.14159292035, 1183.18584070796, 1213.61061946902, 1244.03539823008, 1267.69911504424, 1298.1238938053, 1325.16814159292, 1345.45132743362, 1365.73451327433, 1396.15929203539, 1423.203539823, 1450.24778761061, 1480.67256637168, 1504.33628318584, 1538.14159292035, 1561.80530973451, 1592.23008849557, 1619.27433628318, 1642.93805309734, 1663.22123893805, 1686.88495575221, 1700.40707964601, 1713.92920353982, 1727.45132743362, 1740.97345132743, 1764.63716814159, 1788.30088495575, 1811.96460176991, 1832.24778761061, 1845.76991150442, 1866.05309734513, 1882.95575221238, 1913.38053097345, ]

Lsee_force_model = Regression(Lsee, forces, 2)
anotherTempYval = find_yVal(Lsee_force_model,0.318,2)
print(anotherTempYval)