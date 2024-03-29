import numpy as np
from copy import copy
from gui import GUI
from typing import List, Callable, Any


class Perceptron:
    def __init__(self, learning_rate: float, iterations: int, input_shape: int) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.input_shape = input_shape
        self.__weights = self.__initialize_weights()
        self.__champion = self.__weights                        
        self.__champion_life_time = 0
        self.__weight_life_time = 0   

    def __initialize_weights(self) -> np.ndarray:
        return (np.random.rand(self.input_shape + 1) - 0.5) / 100  

    def __update_weights(self, label: float, prediction: int, input_vector: np.ndarray) -> None:    
        self.__weights[1:] += self.learning_rate * (label - prediction) * input_vector 
        self.__weights[0] += self.learning_rate * (label - prediction)

    def train(self, training_data: np.ndarray, labels: np.ndarray) -> None:
        for _ in range(self.iterations):
            i = np.random.randint(0, len(training_data) - 1)
            input_vector, label = training_data[i], labels[i]
            prediction = self.predict(input_vector, weights = self.__weights)
            if label == prediction:
                self.__weight_life_time += 1
            else:
                self.__challenge_champion() 
                self.__update_weights(label, prediction, input_vector)
        self.__challenge_champion() 

    def __challenge_champion(self) -> None:
        if self.__champion_life_time < self.__weight_life_time:
            self.__champion, self.__champion_life_time = copy(self.__weights), self.__weight_life_time
        self.__weight_life_time = 0

    def predict(self, input_vector: np.ndarray, weights: np.ndarray = None) -> int: 
        if weights is None:                                     
            weights = self.__champion
        if (np.dot(input_vector, weights[1:]) + weights[0]) > 0:
            return 1
        else:
            return 0


def f(prediction_number: int, train_set: np.ndarray) -> Perceptron:    
    perceptron_gen = Perceptron(learning_rate=0.01, iterations=1000, input_shape=35)
    labels = np.zeros(len(train_set))
    idx = 3 * prediction_number
    labels[[idx, idx + 1, idx + 2]] = 1
    perceptron_gen.train(train_set, labels)
    return perceptron_gen


def flatten_all_arrays(l: List[np.ndarray]) -> List[np.ndarray]:
    return [single_2d_array.flatten() for single_2d_array in l]


def predict_list(perceptron_list: List[np.ndarray], data: np.ndarray) -> List[int]:
    return [perceptron.predict(input_vector = data) for perceptron in perceptron_list]


def print_accuracy(prediction_list: List[int], true_labels: np.ndarray) -> None:
    print(np.average(prediction_list == true_labels))


def get_ones_indeces(prediction_list: List[int]) -> str: 
    result = np.where(prediction_list)[0]
    if len(result) == 1:
        return f'predicted value is {result[0]}'
    elif len(result) == 0:
        return f'Number is not recognized'
    else:
        result = [str(number) for number in result]
        return f'predicted values are {", ".join(result)}'


if __name__ == '__main__':
    test_2d_arrays = [
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        ),
        np.array(
            [
                [0.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 1.0, 0.0],
            ]
        ),
        np.array(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ]
        ),
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ),
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ),
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0]
            ]
        ),
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0]
            ]
        ),
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0]
            ]
        ),
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0]
            ]
        ),
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0]
            ]
        ),
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0]
            ]
        ),
        np.array(
            [
                [0.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 1.0, 1.0]
            ]
        ),
        np.array(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0]
            ]
        ),
        np.array(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0]
            ]
        ),
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0]
            ]
        ),
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0]
            ]
        ),
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0]
            ]
        ),
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0]
            ]
        ),
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0]
            ]
        ),
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0]
            ]
        ),
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 1.0, 1.0]
            ]
        ),
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]
            ]
        ),
        np.array(
            [
                [0.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]
            ]
        ),
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]
            ]
        ),
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0]
            ]
        ),
        np.array(
            [
                [0.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 1.0, 0.0]
            ]
        ),
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0]
            ]
        ),
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0]
            ]
        ),
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0]
            ]
        ),
        np.array(
            [
                [0.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0]
            ]
        ),
    ]
    train_set = flatten_all_arrays(test_2d_arrays)
    perceptron_list = [f(i, train_set) for i in range(10)]
    gui = GUI(lambda x: get_ones_indeces(predict_list(perceptron_list, x))) 
