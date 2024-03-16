import numpy as np
from gui import GUI
import os
import struct
from typing import Tuple, List


def load_mnist(path: str, kind: str ='train', shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.int8)
    with open(images_path, 'rb') as imgpath:
        struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        images = images / 255 
    if shuffle:
        data = list(zip(images, labels))
        np.random.shuffle(data)
        data = np.array(data)
        images, labels = data[:,0], data[:,1]
    return images, labels.astype(np.float)

   
class Perceptron:
    def __init__(self, learning_rate: float, max_iterations: int, input_shape: int, min_error: float =1e-9):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.input_shape = input_shape
        self.__weights = self.__initialize_weights()
        self.min_error = min_error
        self.__low_error_counter = 0

    def __initialize_weights(self):
        self.__weights = (np.random.rand(self.input_shape + 1) - 0.5) / 50

    def __calculate_error(self, input_vector: np.ndarray, label: float) -> float:
        return np.sum((self.predict_proba(input_vector) - label) ** 2)  

    def __update_weights(self, label: float, prediction: float, input_vector: np.ndarray) -> None:
        # label - C | prediction - O | input_vector - e 
        self.__weights[1:] += self.learning_rate * (label - prediction) * input_vector
        self.__weights[0] += self.learning_rate * (label - prediction)

    def __activate(self, input_vector: np.ndarray) -> float:
        return 1 / (1 + np.exp(-input_vector))

    def __get_random_example(self, training_data: np.ndarray, labels: np.ndarray) -> Tuple [np.ndarray, float]:  
        idx = np.random.randint(0, len(training_data))  
        return training_data[idx], labels[idx]

    def __continue_learning(self, error: float, counter: int) -> bool:
        if counter > self.max_iterations:
            print(f'I have exceeded the maximum number of iterations: {counter}')
            return False
        if error < self.min_error:
            self.__low_error_counter += 1
            if self.__low_error_counter >= 5:
                print(f'Error is sufficiently small.: {counter}')
                return False
        return True


    def train(self, training_data: np.ndarray, labels: np.ndarray) -> None:
        self.__initialize_weights()
        error = np.inf
        counter = 0
        while self.__continue_learning(error, counter):
            example_td, example_l = self.__get_random_example(training_data, labels)
            output = self.predict_proba(example_td)
            self.__update_weights(label=example_l, prediction=output, input_vector=example_td)
            # error = self.__calculate_error(example_td, example_l)  TODO: Optional functionality
            counter += 1
            # if not self.__continue_learning(error, counter):
            #     break


    def predict(self, input_vector: np.ndarray) -> int:
        proba = self.predict_proba(input_vector)
        if proba > 0.5:
            return 1
        else:
            return 0

    def predict_proba(self, input_vector: np.ndarray) -> float:
        return self.__activate(np.dot(input_vector, self.__weights[1:]) + self.__weights[0])


def train_perceptron(prediction_number: int, train_images: np.ndarray, train_labels: np.ndarray) -> Perceptron:
    perceptron_gen = Perceptron(learning_rate=0.1, max_iterations=120000, input_shape=784, min_error=1e-9)  # 10^7
    perceptron_gen.train(train_images, train_labels == prediction_number)
    return perceptron_gen


def flatten_all_arrays(l: List[np.ndarray]) -> List[np.ndarray]:
    return [single_2d_array.flatten() for single_2d_array in l]


def predict_list(perceptron_list: List[Perceptron], data: np.ndarray) -> int:
    probas_list = [perceptron.predict_proba(data) for perceptron in perceptron_list]
    return np.argmax(probas_list)


def get_confusion_matrix(prediction_list: List[int], true_labels: np.ndarray) -> Tuple[int, int, int, int]:
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    for i in range(len(prediction_list)):
        prediction = prediction_list[i]
        label = true_labels[i]
        if not prediction:
            if label:
                false_negatives += 1
            else:
                true_negatives += 1
        if prediction:
            if label:
                true_positives += 1
            else:
                false_positives += 1

    return true_positives, true_negatives, false_positives, false_negatives


def get_recall(true_positives: int, false_negatives: int) -> float:
    return true_positives / (true_positives + false_negatives)


def get_precision(true_positives: int, false_positives: int) -> float:
    return true_positives / (true_positives + false_positives)


def print_accuracy(prediction_list: List[int], true_labels: np.ndarray) -> float:
    return np.average(prediction_list == true_labels)


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
    x, y = load_mnist('mnist', kind='train') # X_train images, y_train - labels
    X_train, X_test = x[:48000], x[48000:]
    y_train, y_test = y[:48000], y[48000:]
    perceptron_list = [train_perceptron(i, X_train, y_train) for i in range(10)]
    predictions = np.array([predict_list(perceptron_list, image) for image in X_test])
    print(f'Accuracy: {print_accuracy(predictions, y_test)}')
    true_positives, true_negatives, false_positives, false_negatives = get_confusion_matrix(prediction_list=predictions, true_labels=y_test)
    print(f'CONFUSION MATRIX')
    print(f'{true_positives:.2f} | {false_positives:.2f}')
    print(f'{false_negatives:.2f} | {true_negatives:.2f}')
    pre = (true_positives / (true_positives + false_positives))
    rec = (true_positives / (true_positives + false_negatives))
    