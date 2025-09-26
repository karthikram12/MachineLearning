import numpy as np

class LinearReg:
    def __init__(self):
        self.beta = None

    def train(self, x_arr, y_arr):
        if len(x_arr.shape) == 1:
            x_arr = x_arr.reshape(-1, 1)
        self.X = np.column_stack((np.ones(x_arr.shape[0]), x_arr.T))
        self.XTX = self.X.T @ self.X
        self.inv_XTX = np.linalg.inv(self.XTX)
        self.beta = self.inv_XTX @ self.X.T @ y_arr
        self.y_hat = self.X @ self.beta

    def predict(self, x_value):
        self.x_test = np.array(x_value)
        if len(self.x_test.shape) == 1:
            self.x_test = self.x_test.reshape(1, -1)
        self.x_test = np.column_stack((np.ones(self.x_test.shape[0]), self.x_test))
        self.y_test_pred = self.x_test @ self.beta
        return self.y_test_pred
