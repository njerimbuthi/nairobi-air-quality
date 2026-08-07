import numpy as np


def predict(X, theta):
    return np.dot(X, theta)


def compute_cost(X, y, theta):
    prediction = predict(X, theta)
    residual = (prediction - y) ** 2
    return (np.sum(residual) / len(y)) / 2


X = np.array([[1, 25, 60], [1, 30, 80], [1, 28, 70]])

y = np.array([35, 50, 42])

theta = np.array([5, 0.8, 0.2])

print("Predictions:", predict(X, theta))
print("Cost:", compute_cost(X, y, theta))
