import numpy

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
  def __init__(self, data: tuple = None):
    if data == None:
			self.weights = None
      self.weightsSize = 0
		else:
			self.weights = data[0]
			self.weightsSize = data.shape[1]

  def mean(self, ):
    pass

  def cost():
    pass

  def train():
    pass

  def predict():
    pass
