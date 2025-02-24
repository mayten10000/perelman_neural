import random
import re

import numpy as np
import pandas as pd
from pandas import *

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)


def parse_equation(equation):

    match = re.match(r"([+-]?\d*)x\s*([+-]?\s*\d*)\s*=\s*([+-]?\d+)", equation)
    if not match:
        raise ValueError(f"Неправильный формат уравнения: {equation}")

    a = int(match.group(1)) if match.group(1) not in ["", "+", "-"] else (1 if match.group(1) == "+" else -1)
    b = int(match.group(2).replace(" ", "")) if match.group(2) else 0
    c = int(match.group(3))

    return [a, b], c


def load_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    data, labels = [], []

    for eq in df["equation"]:
        x_values, result = parse_equation(eq)
        data.append(x_values)
        labels.append(result)

    return np.array(data, dtype=float), np.array(labels, dtype=float)

class MathNN:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.lr = lr

    def forward(self, x):
        self.hidden = relu(np.dot(x, self.weights1))
        self.output = np.dot(self.hidden, self.weights2)
        return self.output

    def backward(self, x, y, y_pred):
        error = y_pred - y.reshape(-1, 1)
        d_weights2 = np.dot(self.hidden.T, error)
        d_hidden = np.dot(error, self.weights2.T) * relu_derivative(self.hidden)
        d_weights1 = np.dot(x.T, d_hidden)

        self.weights1 -= self.lr * d_weights1
        self.weights2 -= self.lr * d_weights2

        return np.mean(error ** 2)

    def train(self, x, y, epochs=1000):
        for epoch in range(epochs):
            y_pred = self.forward(x)
            loss = self.backward(x, y, y_pred)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")



train_data, train_labels = load_data_from_csv("train_equations.csv")
test_data, test_labels = load_data_from_csv("test_equations.csv")

model = MathNN(input_size=2, hidden_size=16, output_size=1)
model.train(train_data, train_labels)

predictions = model.forward(test_data).squeeze()
error = np.mean(np.abs(predictions - test_labels))


print(f"Средняя ошибка на тесте: {error}")
# Тестирование
predictions = model.forward(test_data).squeeze()
error = np.mean(np.abs(predictions - test_labels))
print(f"Средняя ошибка на тесте: {error}")
