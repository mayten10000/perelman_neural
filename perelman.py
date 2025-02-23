import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers



leaky_relu = lambda x: np.maximum(0.01 * x, x)  # Leaky ReLU
relu_derivative = lambda x: (x > 0).astype(np.float32) + (x <= 0) * 0.01  # Производная для Leaky ReLU


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Вычитаем max для числовой стабильности
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class NNdigitsCV:
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.2):
        # Инициализация весов с использованием He для скрытого слоя и Xavier для выходного
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
        self.dropout_rate = dropout_rate
        # В конструкторе
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

    def forward(self, X, training=True):
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = leaky_relu(self.hidden_layer_input)

        if training:
            # Dropout: случайное отключение нейронов в скрытом слое
            dropout_mask = np.random.rand(*self.hidden_layer_output.shape) > self.dropout_rate
            self.hidden_layer_output *= dropout_mask
        else:
            # Масштабируем скрытые выходы при тестировании
            self.hidden_layer_output /= (1 - self.dropout_rate)

        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output = softmax(self.output_layer_input)
        return self.output

    def backward(self, X, y, learning_rate, visual=False, l2_lambda=0.01):
        output_error = y - self.output
        output_delta = self.output - y  # Градиент softmax + cross-entropy
        hidden_layer_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_layer_delta = hidden_layer_error * relu_derivative(self.hidden_layer_input)

        # Добавление L2 регуляризации (штраф на веса)
        l2_regularization_input_hidden = l2_lambda * self.weights_input_hidden
        l2_regularization_hidden_output = l2_lambda * self.weights_hidden_output

        # Обновление весов с учетом регуляризации
        self.weights_hidden_output -= (self.hidden_layer_output.T.dot(
            output_delta) + l2_regularization_hidden_output) * learning_rate
        self.bias_output -= np.sum(output_delta, axis=0, keepdims=True) * learning_rate

        self.weights_input_hidden -= (X.T.dot(hidden_layer_delta) + l2_regularization_input_hidden) * learning_rate
        self.bias_hidden -= np.sum(hidden_layer_delta, axis=0, keepdims=True) * learning_rate

        if visual:
            print(f"Среднее изменение весов скрытого слоя: {np.mean(np.abs(self.weights_input_hidden))}")
            print(f"Среднее изменение весов выходного слоя: {np.mean(np.abs(self.weights_hidden_output))}")
        print(f"Средний градиент скрытого слоя: {np.mean(np.abs(hidden_layer_delta))}")
        print(f"Средний градиент выходного слоя: {np.mean(np.abs(output_delta))}")

    def train(self, X, y, epochs, learning_rate, visual=False):
        for i in range(epochs):
            self.forward(X, training=True)
            self.backward(X, y, learning_rate, visual)

            if i % 10 == 0:  # Печать каждую 10-ю эпоху
                output = self.output
                print(f"Эпоха {i} - Прогноз: {np.argmax(output, axis=1)[:5]}")
                print(f"Реальные метки: {np.argmax(y, axis=1)[:5]}")

    def save_weights(self, path):
        np.savez(path,
                 weights_input_hidden=self.weights_input_hidden,
                 weights_hidden_output=self.weights_hidden_output,
                 bias_hidden=self.bias_hidden,
                 bias_output=self.bias_output)
        print(f"Модель успешно сохранена по пути {path}")

    def load_weights(self, path):
        data = np.load(path)
        self.weights_input_hidden = data['weights_input_hidden']
        self.weights_hidden_output = data['weights_hidden_output']
        self.bias_hidden = data['bias_hidden']
        self.bias_output = data['bias_output']
        print(f"Модель успешно загружена из пути {path}")


(x_train, y_train), (x_test, y_test) = mnist.load_data()



# Нормализация
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape((60000, 784))
x_test = x_test.reshape((10000, 784))

# One-hot encoding
y_train_one_hot = np.zeros((y_train.size, 10))
y_train_one_hot[np.arange(y_train.size), y_train] = 1

y_test_one_hot = np.zeros((y_test.size, 10))
y_test_one_hot[np.arange(y_test.size), y_test] = 1

# Параметры сети
input_size = 784
hidden_size = 512
output_size = 10

# Создание и обучение модели
nn0 = NNdigitsCV(input_size, hidden_size, output_size)
nn0.train(x_train, y_train_one_hot, epochs=50, learning_rate=0.00001, visual=True)

# Сохранение и загрузка весов
nn0.save_weights('model_cv_digs.npz')
nn0.load_weights('model_cv_digs.npz')

# Прогнозирование и визуализация
for i in range(25):
    sample_input = x_test[i].reshape(1, 784)
    prediction = nn0.forward(sample_input, training=False)  # Выключаем Dropout
    predict_class = np.argmax(prediction)

    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.title(f'Предсказание: {predict_class}\nРеальность: {y_test[i]}')
    plt.show()  # Отображаем результат
