import numpy as np
import matplotlib.pyplot as plt
from numpy import random

x_arr = np.sort(random.randint(10, 20, size=5))
y_arr = np.sort(random.randint(10, 30, size=5))
x_arr_norm = (x_arr - x_arr.mean()) / x_arr.std()
y_arr_norm = (y_arr - y_arr.mean()) / y_arr.std()
r_score = np.sum(x_arr_norm * y_arr_norm) / len(x_arr)
slope = r_score * (y_arr.std() / x_arr.std())
intercept = y_arr.mean() - (slope * x_arr.mean())
y_pred = (slope * x_arr) + intercept

plt.scatter(x_arr, y_arr, color='red')
plt.plot(x_arr, y_pred, color='blue')
plt.show()






