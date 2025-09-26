import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import LinearReg

random.seed(42)

x_arr = np.sort(random.randint(10, 30, size=(5, 5)))
y_arr = np.sort(random.randint(10, 50, size=5))
print(x_arr, y_arr)
linear = LinearReg.LinearReg()
linear.train(x_arr, y_arr)
prediction_values = [[10, 12, 14, 16, 18], [10, 14, 17, 21, 25]]
output = linear.predict(prediction_values)
print(output)

if x_arr.shape == 1:
    plt.scatter(x_arr, y_arr, color='red')
    plt.plot(prediction_values, output, color='blue')
    plt.xlabel("Car Engine Size")
    plt.ylabel("Speed")
    for xi, yi in zip(prediction_values, output):
        plt.vlines(xi, ymin=0, ymax=yi, linestyles='dashed')
        plt.hlines(yi, xmin=0, xmax=xi, linestyles='dashed')

else:
    plt.scatter(y_arr[:len(output)], output)
    plt.plot([y_arr.min(), y_arr.max()],[y_arr.min(), y_arr.max()], color='red')

plt.show()




