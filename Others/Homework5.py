import numpy as np
def load_data(file_path):
    data = np.loadtxt(file_path, delimiter='\t')
    return data


def split_data(data):
    ok = False
    train_data = None
    test_data = None
    while not ok:
        np.random.shuffle(data)
        split_index = int(0.8 * len(data))
        train_data = data[:split_index]
        test_data = data[split_index:]
        train_labels = np.unique(train_data[:, -1])
        if all(label in train_labels for label in train_labels):
            ok = True
    return train_data, test_data


class NeuralNetwork:

    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

        self.input_layer_size = len(self.train_data[0]) - 1
        self.hidden_layer_1_size = 5
        self.hidden_layer_2_size = 4
        self.output_layer_size = len(np.unique(self.train_data[:, -1]))

        self.learning_rate = 0.3
        self.max_epochs = 500

        self.weights_input_hidden_1 = None
        self.weights_hidden_1_hidden_2 = None
        self.weights_hidden_2_output = None
        self.assign_weights()

        self.bias_hidden_1 = None
        self.bias_hidden_2 = None
        self.bias_output = None

        self.new_weights_input_hidden_1 = np.zeros((self.input_layer_size, self.hidden_layer_1_size))
        self.new_weights_hidden_1_hidden_2 = np.zeros((self.hidden_layer_1_size, self.hidden_layer_2_size))
        self.new_weights_hidden_2_output = np.zeros((self.hidden_layer_2_size, self.output_layer_size))

        self.new_bias_hidden_1 = np.zeros((1, self.hidden_layer_1_size))
        self.new_bias_hidden_2 = np.zeros((1, self.hidden_layer_2_size))
        self.new_bias_output = np.zeros((1, self.output_layer_size))

        self.assign_biases()

    def assign_weights(self):
        ok = False
        while not ok:
            self.weights_input_hidden_1 = np.random.uniform(-0.1, 0.1,
                                                            (self.input_layer_size, self.hidden_layer_1_size))
            self.weights_hidden_1_hidden_2 = np.random.uniform(-0.1, 0.1,
                                                               (self.hidden_layer_1_size, self.hidden_layer_2_size))
            self.weights_hidden_2_output = np.random.uniform(-0.1, 0.1,
                                                             (self.hidden_layer_2_size, self.output_layer_size))
            if not (
                    self.weights_input_hidden_1.any() == 0 or self.weights_hidden_1_hidden_2.any() == 0 or self.weights_hidden_2_output.any() == 0):
                ok = True

    def assign_biases(self):
        ok = False
        while not ok:
            self.bias_hidden_1 = np.random.uniform(-0.1, 0.1, (1, self.hidden_layer_1_size))
            self.bias_hidden_2 = np.random.uniform(-0.1, 0.1, (1, self.hidden_layer_2_size))
            self.bias_output = np.random.uniform(-0.1, 0.1, (1, self.output_layer_size))
            if not (np.any(self.bias_hidden_1) == 0 or np.any(self.bias_hidden_2) == 0 or np.any(
                    self.bias_output) == 0):
                ok = True

    def print_parameters(self):
        print("==== Neural Network Parameters ====")
        print(f"Input Layer Size: {self.input_layer_size}")
        print(f"Hidden Layer 1 Size: {self.hidden_layer_1_size}")
        print(f"Hidden Layer 2 Size: {self.hidden_layer_2_size}")
        print(f"Output Layer Size: {self.output_layer_size}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Maximum Epochs: {self.max_epochs}")
        print("\n==== Weights ====")
        print(f"Input-Hidden 1 Weights:\n{self.weights_input_hidden_1}")
        print(f"Hidden 1-Hidden 2 Weights:\n{self.weights_hidden_1_hidden_2}")
        print(f"Hidden 2-Output Weights:\n{self.weights_hidden_2_output}")
        print("\n==== Biases ====")
        print(f"Hidden 1 Bias:\n{self.bias_hidden_1}")
        print(f"Hidden 2 Bias:\n{self.bias_hidden_2}")
        print(f"Output Bias:\n{self.bias_output}")

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def bipolar_sigmoid(self, x):
        return (1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x))

    def bipolar_sigmoid_derivative(self, x):
        return 1 - self.bipolar_sigmoid(x) ** 2

    def error_function(self, target, output):
        return 0.5 * np.mean((target - output) ** 2)

    def feedforward(self, input_data):
        hidden_layer_1_input = np.dot(input_data, self.weights_input_hidden_1) + self.bias_hidden_1
        hidden_layer_1_output = self.sigmoid(hidden_layer_1_input)

        hidden_layer_2_input = np.dot(hidden_layer_1_output, self.weights_hidden_1_hidden_2) + self.bias_hidden_2
        hidden_layer_2_output = self.sigmoid(hidden_layer_2_input)

        output_layer_input = np.dot(hidden_layer_2_output, self.weights_hidden_2_output) + self.bias_output
        output_layer_output = self.sigmoid(output_layer_input)

        return output_layer_output, hidden_layer_1_output, hidden_layer_2_output

    def backpropagation(self, input_data, target):
        # Realizează feedforward pentru a obține ieșirile fiecărui strat
        output_layer_output, hidden_layer_1_output, hidden_layer_2_output = self.feedforward(input_data)

        # Calculează eroarea și delta pentru stratul de ieșire
        output_layer_error = target - output_layer_output
        output_layer_delta = self.sigmoid_derivative(output_layer_output) * output_layer_error

        # Propagă înapoi eroarea și calculează delta pentru straturile ascunse
        hidden_layer_2_error = np.dot(output_layer_delta, self.weights_hidden_2_output.T)
        hidden_layer_2_delta = self.sigmoid_derivative(hidden_layer_2_output) * hidden_layer_2_error

        hidden_layer_1_error = np.dot(hidden_layer_2_delta, self.weights_hidden_1_hidden_2.T)
        hidden_layer_1_delta = self.sigmoid_derivative(hidden_layer_1_output) * hidden_layer_1_error

        # Actualizează greutățile și bias-urile folosind delta-urile calculate
        # Matricea pentru greutățile dintre stratul ascuns 2 și stratul de ieșire
        self.new_weights_hidden_2_output += self.learning_rate * np.dot(hidden_layer_2_output.T.reshape(-1, 1),
                                                                        output_layer_delta.reshape(1, -1))
        # Matricea pentru greutățile dintre stratul ascuns 1 și stratul ascuns 2
        self.new_weights_hidden_1_hidden_2 += self.learning_rate * np.dot(hidden_layer_1_output.T.reshape(-1, 1),
                                                                          hidden_layer_2_delta.reshape(1, -1))
        # Matricea pentru greutățile dintre stratul de intrare și stratul ascuns 1
        self.new_weights_input_hidden_1 += self.learning_rate * np.dot(input_data.reshape(-1, 1),
                                                                       hidden_layer_1_delta.reshape(1, -1))

        # Vectorul pentru bias-ul stratului de ieșire
        self.new_bias_output += self.learning_rate * output_layer_delta
        # Vectorul pentru bias-ul stratului ascuns 2
        self.new_bias_hidden_2 += self.learning_rate * hidden_layer_2_delta
        # Vectorul pentru bias-ul stratului ascuns 1
        self.new_bias_hidden_1 += self.learning_rate * hidden_layer_1_delta

        # Returnează eroarea stratului de ieșire
        return output_layer_error

    def train(self):
        total_error = 0
        for epoch in range(self.max_epochs):
            epoch_error = 0
            for input_data in self.train_data:
                target = np.zeros(self.output_layer_size)
                label = int(input_data[-1])
                label_index = label - 1
                target[label_index] = 1
                output_layer_error = self.backpropagation(input_data[:-1], target)
                epoch_error += self.error_function(target, output_layer_error)

            self.update_weights_and_biases()


            epoch_mean_error = epoch_error / len(self.train_data)
            total_error += epoch_mean_error
            if epoch % 100 == 0:
                print(f'Epoch: {epoch}, error: {epoch_mean_error}')


        average_error = total_error / self.max_epochs
        print(f'Average error over all epochs: {average_error}')

    def update_weights_and_biases(self):
        self.weights_input_hidden_1 += self.new_weights_input_hidden_1
        self.weights_hidden_1_hidden_2 += self.new_weights_hidden_1_hidden_2
        self.weights_hidden_2_output += self.new_weights_hidden_2_output

        self.bias_output += self.new_bias_output
        self.bias_hidden_2 += self.new_bias_hidden_2
        self.bias_hidden_1 += self.new_bias_hidden_1

        self.new_weights_input_hidden_1 = 0
        self.new_weights_hidden_1_hidden_2 = 0
        self.new_weights_hidden_2_output = 0

        self.new_bias_hidden_1 = 0
        self.new_bias_hidden_2 = 0
        self.new_bias_output = 0

    def run_test(self):
        correct_predictions = 0
        total_instances = len(self.test_data)

        for input_data in self.test_data:
            true_label = int(input_data[-1])
            predicted_label = self.predict(input_data[:-1])
            if predicted_label == true_label:
                correct_predictions += 1

        accuracy_percentage = (correct_predictions / total_instances) * 100
        print(f"Accuracy: {correct_predictions} / {total_instances} ({accuracy_percentage:.2f}%)")

    def predict(self, input_data):
        output_layer_output = self.feedforward(input_data)[0]
        prediction = np.argmax(output_layer_output) + 1
        return prediction


data = load_data('../Lab5/seets_dataset.txt')
train_data, test_data = split_data(data)
nm = NeuralNetwork(train_data, test_data)
# nm.print_parameters()

nm.train()
nm.run_test()

