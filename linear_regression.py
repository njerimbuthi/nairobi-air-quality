import numpy as np


def predict(X, theta):
    return np.dot(X, theta)


def compute_cost(X, y, theta):
    prediction = predict(X, theta)
    residual = (prediction - y) ** 2
    return (np.sum(residual) / len(y)) / 2


def gradient_descent(X, y, theta, alpha, iterations):
    for i in range(iterations):
        prediction = predict(X, theta)
        residual = prediction - y
        gradient = np.dot(X.T, residual) / len(y)
        theta = theta - alpha * gradient
    return theta


X = np.array([[1, 25, 60], [1, 30, 80], [1, 28, 70]])

y = np.array([35, 50, 42])

theta = np.array([0.0, 0.0, 0.0])

print("Before training:")
print("Theta:", theta)
print("Cost:", compute_cost(X, y, theta))

theta = gradient_descent(X, y, theta, alpha=0.0001, iterations=1000)

print("\nAfter training:")
print("Theta:", theta)
print("Cost:", compute_cost(X, y, theta))
print("Predictions:", predict(X, theta))
print("Actual:", y)
