import numpy as np

"""
hypotesis:
		h(x1, x1, ..., xN) = o1.x1 + o2.x2 + ... + oN.xN

cost function:
		J(o1, o2, ..., oN) = (1/2N) * sum((h(x) - y)^2)

"""

"""
		C1 C2 C3 C4
R0  10 20 30 40
R1  11 21 31 31
R2  12 22 32 42
R3  13 23 33 43
R4  14 24 34 44
"""


class LinearRegression():
    def __init__(self, data=None, scaling=False, normalization=False):
        self.weights = list()
        self.weights_size = data.shape[1] if data else 0
        self.X = data[0] if data else None
        self.y = data[1] if data else None
        self.data_scaling(X, scaling, normalization)

    def data_scaling(X, scaling, normalization):
        if normalization:
            for col_i in range(X.shape[1]):
                col = X[:, col_i]
                col_mean = np.mean(col)
                col_range = max(col) - min(col)
                for row_i, row in enumerate(X):
                    X[row_i][col_i] = (
                        (X[row_i][col_i] - col_mean) / col_range) * 1.0
        elif scaling:
            for col_i in range(X.shape[1]):
                col = X[:, col_i]
                col_range = max(col) - min(col)
                for row_i, row in enumerate(X):
                    X[row_i][col_i] = (X[row_i][col_i] / col_range) * 1.0
