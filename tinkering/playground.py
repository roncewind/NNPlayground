import numpy as np

input_vector = np.array([1.66, 1.56])
weights_1 = np.array([1.45, -0.66])
bias = np.array([0.0])


# -----------------------------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# -----------------------------------------------------------------------------
def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))


# -----------------------------------------------------------------------------
def make_prediction(input_vector, weights, bias):
    layer_1 = np.dot(input_vector, weights) + bias
    layer_2 = sigmoid(layer_1)
    return layer_2


# -----------------------------------------------------------------------------
prediction = make_prediction(input_vector, weights_1, bias)
print(f"Prediction: {prediction}")

input_vector = np.array([2, 1.5])

prediction = make_prediction(input_vector, weights_1, bias)

print(f"The prediction result is: {prediction}")
target = 0
mse = np.square(prediction - target)
print(f"Prediction: {prediction}; Error: {mse}")
derivative = 2 * (prediction - target)
print(f"The derivative is {derivative}")

weights_1 = weights_1 - derivative
prediction = make_prediction(input_vector, weights_1, bias)
error = (prediction - target) ** 2
print(f"Prediction: {prediction}; Error: {error}")

derror_dprediction = 2 * (prediction - target)
layer_1 = np.dot(input_vector, weights_1) + bias
dprediction_dlayer1 = sigmoid_deriv(layer_1)
dlayer1_dbias = 1

derror_dbias = derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
