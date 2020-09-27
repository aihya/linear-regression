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
import math


class LinearRegression():
    def __init__(self, data=None, scaling=False, normalization=False):
        self.weights = None
        self.X = np.array(data[0], dtype=float) if data else None
        self.y = np.array(data[1], dtype=float) if data else None
        self.data_scaling(self.X, scaling, normalization)

    def data_scaling(self, X, scaling, normalization):
        if normalization:
            for col_i in range(X.shape[1]):
                col = X[:, col_i]
                col_mean = np.mean(col)
                col_range = max(col) - min(col)
                for row_i, row in enumerate(X):
                    X[row_i][col_i] = (X[row_i][col_i] - col_mean) / col_range
        elif scaling:
            for col_i in range(X.shape[1]):
                col = X[:, col_i]
                col_range = max(col) - min(col)
                for row_i, row in enumerate(X):
                    X[row_i][col_i] = X[row_i][col_i] / col_range

    def random_weights(self, X):
        return np.empty([1, X.shape[1]], dtype=float)

    def hypothesis(self, X, weights):
        _X = np.append(X.T, np.ones([1, weights.length])).T
        return np.matmul(_X, weights, dtype=float)

    def cost(self, X, y, weights, length):
        _hypothesis = self.hypothesis(X, weights)
        _sum = 0
        for i in range(length):
            _sum = _sum + math.pow(_hypothesis[i] + y[i], 2)
        return 1/(2*length) * _sum

    # to-do
    def gradient_descent(self, X, y, weights, alpha):
        pass

    def train(self, learning_rate=0.1):
        pass
