
import operator

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


x = np.transpose([-29.9193548387096, -25.3225806451612, -23.1451612903225, -21.4516129032258, -19.2741935483871, -17.3387096774193, -15.6451612903225, -13.9516129032258, -12.5, -10.3225806451612, -8.14516129032258, -5.72580645161291, -3.30645161290323, 0.0806451612903202, 2.74193548387096, 9.51612903225805, 11.9354838709677, 14.5967741935483, 17.9838709677419, 21.6129032258064, 23.5483870967741, 25.9677419354838, 27.4193548387096, 29.8387096774193, 19.6774193548386, 7.09677419354838, 4.91935483870967, -27.258064516129])
y = np.transpose([71.4117647058823, 70, 68.1176470588235, 66.235294117647, 63.4117647058823, 61.0588235294117, 58.235294117647, 55.8823529411764, 53.5294117647059, 50.235294117647, 47.4117647058823, 45.0588235294117, 43.6470588235294, 42.235294117647, 41.7647058823529, 40.3529411764706, 39.4117647058823, 38, 36.1176470588235, 33.2941176470588, 31.4117647058823, 29.0588235294117, 27.6470588235294, 25.2941176470588, 34.7058823529411, 40.8235294117647, 41.2941176470588, 70.9411764705882])


# transforming the data to include another axis
x = x[:, np.newaxis]
y = y[:, np.newaxis]

polynomial_features= PolynomialFeatures(degree=5)
x_poly = polynomial_features.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
r2 = r2_score(y,y_poly_pred)
print(rmse)
print(r2)

plt.scatter(x, y, s=10)
# sort the values of x before line plot
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)
x, y_poly_pred = zip(*sorted_zip)
plt.plot(x, y_poly_pred, color='m')
plt.show()
