import numpy as np
from sklearn.datasets import load_digits
from gui import GUI
import os
import struct
from sklearn.metrics import recall_score, precision_score, roc_auc_score, f1_score, accuracy_score
from typing import List
import pandas as pd
from mnist import MNIST
from pandas import read_csv
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Any


def logit(vector: np.ndarray) -> np.ndarray: 
    return 1 / (1 + np.exp(-vector))  


def logit_derivative(vector: np.ndarray) -> np.ndarray: 
    return logit(vector) * (1 - logit(vector))
    

class Perceptron:
    def __init__(self, learning_rate: float, max_iterations: int, input_shape: int, min_error: float = 1e-9):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.input_shape = input_shape
        self.weights = self.__initialize_weights()
        self.min_error = min_error
        self.__low_error_counter = 0

    def __initialize_weights(self) -> np.ndarray:
        return np.random.uniform(-1, 1, self.input_shape + 1)

    def activate(self, input_vector: np.ndarray) -> np.ndarray:
        return logit(input_vector)  

    def predict(self, input_vector: np.ndarray) -> int: 
        proba = self.predict_proba(input_vector)
        if proba > 0.5:
            return 1
        else:
            return 0

    def predict_proba(self, input_vector: np.ndarray) -> float: 
        return self.activate(self.multiply_by_weights(input_vector))


    def multiply_by_weights(self, input_vector: np.ndarray) -> float: 
        return np.dot(input_vector, self.weights[1:]) + self.weights[0]

class Layer:
    def __init__(self, input_shape: int, output_shape: int, learning_rate: float, max_iterations: int): 
        self.perceptron_list = [Perceptron(input_shape=input_shape, learning_rate=learning_rate, max_iterations=max_iterations) for _ in range(output_shape)]
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.delta = None

    def update_weights(self, eta: float, x: np.ndarray) -> None: 
        substrahend = eta * self.delta.T.dot(x) 
        for i, perceptron in enumerate(self.perceptron_list):
            perceptron.weights[1:] -= substrahend[i] 

    @property
    def weights(self) -> np.ndarray:
        return np.array([perceptron.weights for perceptron in self.perceptron_list])

    def calculate_derivative(self, param: np.ndarray) -> np.ndarray: 
        return logit_derivative(param) 

    def predict(self, x: np.ndarray) -> np.ndarray: 
        output = [] 
        for perceptron in self.perceptron_list:
            prediction = perceptron.predict_proba(x)
            output.append(prediction)
        return np.array(output).T

    def multiply_by_weights(self, x: np.ndarray) -> np.ndarray:
        output = [] 
        for perceptron in self.perceptron_list:
            prediction = perceptron.multiply_by_weights(x)
            output.append(prediction)
        return np.array(output).T

    def activate(self, vector: np.ndarray) -> np.ndarray: 
        return self.perceptron_list[-1].activate(vector) 


class NeuralNetwork:
    def __init__(self, hidden_layer_sizes: Tuple[int, ...], eta: float, no_iterations: int, tol: float = 1e-5):
        self.eta = eta  
        self.hidden_layer_sizes = hidden_layer_sizes
        self.no_iterations = no_iterations
        self.layers = None 
        self.tol = tol


    def train(self, x: np.ndarray, y: np.ndarray) -> None: 
        self.layers = self.__get_layers(x, y) 
        previous_error = np.inf
        for iteration in range(self.no_iterations):
            if iteration % 2 == 1:
                print(f'{iteration}, loss:{self.loss:.5f}')
            for batch_i in range(int(len(x) / 16)): 
                batch_x = x[16 * batch_i: 16 * batch_i + 16]
                batch_y = y[16 * batch_i: 16 * batch_i + 16]
                outputs = self.__forward(batch_x) 
                self.__backward(batch_y, outputs) 
                error = self.calculate_loss(batch_y, self.layers[-1].activate(outputs[-1]))

                self.layers[0].update_weights(self.eta, batch_x) 
                for i in range(1, len(self.layers)):  
                    layer = self.layers[i] 
                    activation = layer.activate(outputs[i - 1])
                    layer.update_weights(self.eta, activation)
            previous_error = error

    def __should_break(self, error, previous_error):
        loss_drops = error <= previous_error - self.tol
        loss_jumps = error > previous_error * 10
        return not loss_drops and not loss_jumps

    def calculate_loss(self, y_true, y_pred):
        self.loss = np.sqrt(np.sum((y_true - y_pred)**2))
        return self.loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        probas = self.predict_proba(x) 
        return np.argmax(probas, axis=1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        output = self.__forward(x)[-1]
        last_layer = self.layers[-1]
        return last_layer.activate(output)

    def __calculate_hidden_layer_sigma(self, next_layer):
        wk = next_layer.weights[:,1:]
        sigma = next_layer.delta.dot(wk)
        return sigma

    def __get_layers(self, x: np.ndarray, y: np.ndarray) -> List[Layer]:
        layers = []
        input_shape = x.shape[1]
        output_shape = y.shape[1]  
        layers.append(Layer(input_shape=input_shape, output_shape=self.hidden_layer_sizes[
            0], learning_rate=self.eta, max_iterations=self.no_iterations))  
        for i in range(len(self.hidden_layer_sizes) - 1):  
            layers.append(Layer(input_shape=self.hidden_layer_sizes[i], output_shape=self.hidden_layer_sizes[i + 1], learning_rate=self.eta, max_iterations=self.no_iterations))
        layers.append(Layer(input_shape=self.hidden_layer_sizes[-1], output_shape=output_shape, learning_rate=self.eta, max_iterations=self.no_iterations))
        return layers

    def __forward(self, x: np.ndarray) -> List[np.ndarray]:
        layer_outputs = [None for _ in self.hidden_layer_sizes] + [None]
        layer_outputs[0] = self.layers[0].multiply_by_weights(x)
        previous_layer = self.layers[0]
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            previous_layer_activation = previous_layer.activate(layer_outputs[i - 1])
            prediction = layer.multiply_by_weights(previous_layer_activation)
            layer_outputs[i] = prediction
            previous_layer = layer
        return layer_outputs

    def __backward(self, y_train: np.ndarray, layer_outputs: List[np.ndarray]) -> None: 
        layer_outputs = layer_outputs[::-1]
        delta = []
        sigma = self.layers[-1].activate(layer_outputs[0]) - y_train 
        derivative = self.layers[-1].calculate_derivative(layer_outputs[0])
        delta.append(sigma * derivative) 
        self.layers[-1].delta = sigma * derivative
        for i in range(1, len(self.layers)):
            layer = self.layers[::-1][i]
            next_layer = self.layers[::-1][i - 1]
            sigma = self.__calculate_hidden_layer_sigma(next_layer)
            derivative = layer.calculate_derivative(layer_outputs[i])
            layer.delta = sigma * derivative


if __name__ == '__main__':
    data = pd.read_csv('mnist_train.csv')
    y = data['label'].to_numpy()
    x = data.drop('label', axis=1).to_numpy()
    x = x / x.max()
    X_train, X_test, y_train, y_test = train_test_split(x, y)
    y_train = np.array(pd.get_dummies(y_train))

    nn = NeuralNetwork(hidden_layer_sizes=(300,), eta=0.1, no_iterations=30, tol=1e-9)

    nn.train(X_train, y_train)
    predictions = nn.predict(X_test)
    probas = nn.predict_proba(X_test)
    print(
        f' Accuracy: {100 * accuracy_score(y_test, predictions):.2f}%')
    print(
        f' F1_score: {100 * f1_score(y_test, predictions, average="macro"):.2f}%')  
    print(
        f' Recall: {100 * recall_score(y_test, predictions, average="macro"):.2f}%')  
    print(f' Precision: {100 * precision_score(y_test, predictions, average="macro"):.2f}%')
    