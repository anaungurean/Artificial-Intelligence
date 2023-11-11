import numpy as np


class NeuralNetwork:
    def __init__(self, file_path):
        self.data = self.load_data(file_path)
        self.train_data, self.test_data = self.split_data(self.data)

        #parametrii
        self.input_layer_size = self.train_data.shape[1] - 1  #nr atributelor de intrare(fara ultima coloana pentru clasa)
        self.hidden_layer_size = 5 #nr de neuroni
        self.output_layer_size = len(np.unique(self.data[:, -1])) #nr de clase = dim stratului de iesire

        self.learning_rate = 0.1
        self.max_epochs = 500

        #ponderile
        self.weights_input_hidden = np.random.uniform(-0.1, 0.1, (self.input_layer_size, self.hidden_layer_size))
        self.weights_hidden_output = np.random.uniform(-0.1, 0.1, (self.hidden_layer_size, self.output_layer_size))

    def load_data(self, file_path):
        data = np.loadtxt(file_path, delimiter='\t')
        return data

    def split_data(self, data):
        unique_labels = np.unique(data[:, -1])
        ok = False
        train_data = None
        test_data = None
        while not ok:
            print("Shuffling data...")
            np.random.shuffle(data)
            split_index = int(.8 * len(data))
            train_data = data[:split_index]
            test_data = data[split_index:]
            train_labels = np.unique(train_data[:, -1])
            if all(label in train_labels for label in unique_labels):
                ok = True
        return train_data, test_data

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    #f'(x) = f(x) * (1 - f(x))
    def sigmoid_derivative(selfself, x):
        return x * (1 - x)

    def relu(self, x):
        return 0 if x < 0 else x

    def relu_derivative(self, x):
        return 0 if x < 0 else 1

    def bipolar_sigmoid(self, x):
        return (1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x))

    #f′(x) = 1 − (f(x))^2
    def bipolar_sigmoid_derivative(self, x):
        return 1 - x**2

    def error_function(self, target, output):
        return 0.5 * np.mean((target - output) ** 2)

    def print_data(self):
        print(f'Train data is: \n{self.train_data}')
        print(f'Testing data is: \n{self.test_data}')


nm = NeuralNetwork('seets_dataset.txt')
nm.print_data()




