import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
import time

leaky_relu = lambda x: np.maximum(0.01 * x, x)  # Leaky ReLU
relu_derivative = lambda x: (x > 0).astype(np.float32) + (x <= 0) * 0.01  # Производная для Leaky ReLU


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Вычитаем max для числовой стабильности
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class NNdigitsCV:
    
    def __init__(self, input_size=784, hidden_size=128, output_size=10, dropout_rate=0.1):

        # Инициализация весов с использованием He для скрытого слоя и Xavier для выходного
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)

        self.dropout_rate = dropout_rate

        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

    def forward(self, X, training=True):
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = leaky_relu(self.hidden_layer_input)

        self.hidden_layer_output /= (1 - self.dropout_rate)

        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output = softmax(self.output_layer_input)
        return self.output

    def backward(self, X, y, learning_rate, visual=False, l2_lambda=0.01):
        #output_error = y - self.output
        output_delta = self.output - y  # Градиент softmax + cross-entropy
        hidden_layer_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_layer_delta = hidden_layer_error * relu_derivative(self.hidden_layer_input)

        # Добавление L2 регуляризации (штраф на веса)
        l2_regularization_input_hidden = l2_lambda * self.weights_input_hidden
        l2_regularization_hidden_output = l2_lambda * self.weights_hidden_output

        # Обновление весов с учетом регуляризации
        self.weights_hidden_output -= (self.hidden_layer_output.T.dot(output_delta) + l2_regularization_hidden_output) * learning_rate
        self.bias_output -= np.sum(output_delta, axis=0, keepdims=True) * learning_rate

        self.weights_input_hidden -= (X.T.dot(hidden_layer_delta) + l2_regularization_input_hidden) * learning_rate
        self.bias_hidden -= np.sum(hidden_layer_delta, axis=0, keepdims=True) * learning_rate

        if visual:
            print(f"Среднее изменение весов скрытого слоя: {np.mean(np.abs(self.weights_input_hidden))}")
            print(f"Среднее изменение весов выходного слоя: {np.mean(np.abs(self.weights_hidden_output))}")
            print(f"Средний градиент скрытого слоя: {np.mean(np.abs(hidden_layer_delta))}")
            print(f"Средний градиент выходного слоя: {np.mean(np.abs(output_delta))}")

    def train(self, X, y, epochs, learning_rate, visual=False, saving_possibillity=True):
        
        for i in range(epochs):
            self.forward(X, training=True)
            self.backward(X, y, learning_rate, visual)

            if visual:
                output = self.output
                print(f"Эпоха {i+1}/{epochs} пройдена:")
                print(f"\t- Прогноз: {np.argmax(output, axis=1)}")
                print(f"\t- Реальные метки: {np.argmax(y, axis=1)}")

        if saving_possibillity==True:
            
            if input('Хотите ли вы сохранить модель? (Y/n): ').upper() == 'Y':

                nn0.save_weights(name=f'models/{input('Введите имя модели: ')}.npz', visual=True)

            else:

                print('Модель не была сохранена после обучения')
                

    def test(self, x_test, y_test, tryes, visual=False):
        
        lucky_tryes = 0
    
        st = time.time()
        
        for i in range(tryes):
            
            sample_input = x_test[i].reshape(1, 784)
            prediction = self.forward(sample_input, training=False)
            predict_class = np.argmax(prediction)
    
            if (predict_class == y_test[i]).all(): lucky_tryes += 1
            
            if visual:
                plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
                plt.title(f'Предсказание: {predict_class}\nРеальность: {np.argmax(y_test[i] == 1)}') #{y_test[i]}')
                plt.show()  
    
        return round((lucky_tryes / tryes) * 100, 2)

    def long_train(self, cntEpochs, safe_cnt, start_epochs):
    
        apprchs = cntEpochs // safe_cnt
    
        for i in range(apprchs):
            
            nn0.train(x_train, y_train_one_hot, epochs=safe_cnt, learning_rate=0.00001, visual=True)
    
            nn0.save_weights(f'D:/models/model_cv_digs ({start_epochs + safe_cnt * (i + 1)}).npz')

    def total_test(self, start, add, test_tryes):
    
        for m in range(0,26):
            
            nn0.load_weights(f'model_cv_digs ({start + add * m}).npz', visual=False)
            sumOfProbs = sum([nn0.test(10000) for _ in range(test_tryes)])
            print(f'model_cv_digs ({start + add * m}).npz : {round(sumOfProbs / test_tryes, 2)} %')

    def save_weights(self, path='models/model_cv_digs.npz', visual=True):
        np.savez(path,
                 weights_input_hidden=self.weights_input_hidden,
                 weights_hidden_output=self.weights_hidden_output,
                 bias_hidden=self.bias_hidden,
                 bias_output=self.bias_output)
        if visual: print(f"Модель успешно сохранена по пути {path}")

    def load_weights(self, name='models/model_cv_digs (22000).npz', visual=True):
        data = np.load(name)
        self.weights_input_hidden = data['weights_input_hidden']
        self.weights_hidden_output = data['weights_hidden_output']
        self.bias_hidden = data['bias_hidden']
        self.bias_output = data['bias_output']
        if visual: print(f"Модель {name} успешно загружена")

    def prepare_dataset(self, ch=0):
        
        
        if ch == 0:  # MNIST
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

            return x_train, y_train_one_hot, x_test, y_test_one_hot

    def start_model(self, skip=True):

        if skip:
            
            nn0.load_weights()
            return "Запущен стандартный сценарий"
        
        if input("Хотите ли вы загрузить модель (иначе будет создана новая) ? (Y/n): ").upper() == 'Y': 
            
            nn0.load_weights()

        ds = int(input("Выберите датасет:\n- (0) - MNIST ;\n> "))

        if ds == 0:

            x_train, y_train_one_hot, x_test, y_test = nn0.prepare_dataset()
            print(type(y_test))
            
        else:
            print("Некорректный ввод. Программа будет перезапущена...")
            nn0.start_model(skip=False)  
            
        m = int(input("Выберите режим:\n- (0) - Обучение ;\n- (1) - Тестирование ;\n> "))

        if m == 0:

            epochs = int(input("Выберите кол-во эпох обучения: "))
            
            if input("Отображать ли данные обучения ? (Y/n): ").upper() == 'Y':

                nn0.train(x_train, y_train_one_hot, epochs, learning_rate=0.00001, visual=True, saving_possibillity=True)

            else:

                nn0.train(x_train, y_train_one_hot, epochs, learning_rate=0.00001, visual=False, saving_possibillity=True)

        elif m == 1:

            test_data = int(input("Выберите кол-во тестовых изображений: "))

            if input("Отображать ли данные тестирования ? (Y/n): ").upper() == 'Y':

                nn0.test(x_test, y_test, test_data, visual=True)

            else:

                nn0.test(x_test, y_test, test_data, visual=False)

        else:
            print("Некорректный ввод. Программа будет перезапущена...")
            nn0.start_model(skip=False)            

nn0 = NNdigitsCV()
nn0.start_model(skip=False)
