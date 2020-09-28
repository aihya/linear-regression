# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    linear_regression.py                               :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: aihya <aihya@student.1337.ma>              +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/09/27 23:33:23 by aihya             #+#    #+#              #
#    Updated: 2020/09/27 23:33:23 by aihya            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np


class LinearRegression():
    def __init__(self, scaling=False, normalization=False):
        self._scaling = scaling
        self._normalization = normalization

    def data_scaling(self, X, scaling, normalization):
        if normalization:
            for col_i in range(X.shape[1]):
                col = X[:, col_i]
                col_mean = np.mean(col)
                col_range = max(col) - min(col)
                for row_i, _ in enumerate(X):
                    X[row_i][col_i] = (X[row_i][col_i] - col_mean) / col_range
        elif scaling:
            for col_i in range(X.shape[1]):
                col = X[:, col_i]
                col_range = max(col) - min(col)
                for row_i, _ in enumerate(X):
                    X[row_i][col_i] = X[row_i][col_i] / col_range
        return X

    def random_weights(self, size):
        return np.random.rand(size, 1) * np.random.rand(1) * 100

    def hypothesis(self, X, weights):
        return np.matmul(X, weights, dtype=float)

    def cost(self, X, y, weights):
        m = y.shape[0]
        _hypothesis = self.hypothesis(X, weights)
        _sum = 0
        for i in range(m):
            _sum = _sum + np.math.pow(_hypothesis[i][0] - y[i][0], 2)
        return 1/(2*m) * _sum

    def derivative(self, X, y, weights, index):
        m = y.shape[0]
        _hypothesis = self.hypothesis(X, weights)
        _sum = 0
        for i in range(m):
            _sum = _sum + (_hypothesis[i][0] - y[i][0]) * X[i][index]
        return 1/m * _sum

    def gradient_descent(self, X, y, weights, alpha, iterations):
        for _ in range(iterations):
            weights_copy = np.copy(weights)
            for i, _ in enumerate(weights):
                cost_derivative = self.derivative(X, y, weights_copy, i)
                weights[i][0] = weights[i][0] - alpha*cost_derivative
        print("Iteration {} | Cost {}".format(
            _ + 1, self.cost(X, y, weights)))
        return weights

    def fit(self, X, y, lr=0.1, iters=10):
        X = self.data_scaling(X, self._scaling, self._normalization)
        X = np.append(np.ones([X.shape[0], 1], dtype=float), X, axis=1)
        y = y.reshape([y.shape[0], 1])
        _weights = self.random_weights(X.shape[1])
        weights = self.gradient_descent(X, y, _weights, lr, iters)
        return weights.reshape([1, weights.shape[0]])
