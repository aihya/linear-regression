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
	def __init__(self, fit_intercept=True, normalize=False, verbose=False):
		self.normalize = normalize
		self.fit_intercept = fit_intercept
		self.weights = None
		self.intercept = None

	def data_scaling(self, X, normalize):
		if normalize:
			for col_i in range(X.shape[1]):
				col = X[:, col_i]
				col_mean = np.mean(col)
				col_range = max(col) - min(col)
				for row_i, _ in enumerate(X):
					X[row_i][col_i] = (X[row_i][col_i] - col_mean) / col_range
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

	def gradient_descent(self, X, y, weights, alpha, iters):
		for _ in range(iters):
			if self.verbose:
				print("Iteration: {}, Cost {}\r".format(_+1, self.cost(X, y, weights)), end='')
			_diff = self.hypothesis(X, weights)
			weights = weights - (alpha/m) * np.matmul(X.T, _diff, dtype=float) 
		if self.verbose:
			print()
		return weights
		
	def normal_equation(self, X, y):
		centered = np.matmul(X.T, X)
		inversed = np.linalg.pinv(centered)
		weights = np.matmul(inversed, np.matmul(X.T, y))
		return weights

	def fit(self, X, y, lr=0.1, iters=10):
		X = self.data_scaling(X, self.normalize)
		y = y.reshape([y.shape[0], 1])
		if (X.shape[0] >= 1000 or X.shape[1] >= 1000):
			X = np.append(np.ones([X.shape[0], 1], dtype=float), X, axis=1)
			_weights = self.random_weights(X.shape[1])
			weights = self.gradient_descent(X, y, _weights, lr, iters)
		else:
			if self.fit_intercept:
				X = np.append(np.ones([X.shape[0], 1], dtype=float), X, axis=1)
			weights = self.normal_equation(X, y)
			if self.fit_intercept:
				self.intercept = weights[0]
		self.weights = weights.reshape([1, weights.shape[0]])[0]

	def predict(self, X):
		X = np.append(np.ones([X.shape[0], 1], dtype=float), X, axis=1)
		weights = self.weights.reshape([self.weights.shape[1], 1])
		return np.matmul(X, weights).reshape([1, X.shape[0]])
