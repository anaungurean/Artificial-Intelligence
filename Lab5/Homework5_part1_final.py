import numpy as np


def load_data(file_path):
    data = np.loadtxt(file_path, delimiter='\t')
    return data
def split_data(data):
    unique_labels = np.unique(data[:, -1])
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

        self.learning_rate = 0.1
        self.max_epochs = 500

        self.weights_input_hidden_1 = None
        self.weights_hidden_1_hidden_2 = None
        self.weights_hidden_2_output = None
        self.assign_weights()

        self.bias_hidden_1 = None
        self.bias_hidden_2 = None
        self.bias_output = None
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
            if not(self.weights_input_hidden_1.any() == 0 or self.weights_hidden_1_hidden_2.any() == 0 or self.weights_hidden_2_output.any() == 0):
                ok = True

    def assign_biases(self):
        ok = False
        while not ok:
            self.bias_hidden_1 = np.random.uniform(-0.1, 0.1, (1, self.hidden_layer_1_size))
            self.bias_hidden_2 = np.random.uniform(-0.1, 0.1, (1, self.hidden_layer_2_size))
            self.bias_output = np.random.uniform(-0.1, 0.1, (1, self.output_layer_size))
            if not (np.any(self.bias_hidden_1) == 0 or np.any(self.bias_hidden_2) == 0 or np.any(self.bias_output) == 0):
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
        return 1 - self.bipolar_sigmoid(x)**2

    def error_function(self, target, output):
        return 0.5 * np.mean((target - output) ** 2)

    def feedforward(self, input_data):
        hidden_layer_1_input = np.dot(input_data, self.weights_input_hidden_1) + self.bias_hidden_1
        hidden_layer_1_output = self.sigmoid(hidden_layer_1_input)

        hidden_layer_2_input = np.dot(hidden_layer_1_output, self.weights_hidden_1_hidden_2) + self.bias_hidden_2
        hidden_layer_2_output = self.relu(hidden_layer_2_input)

        output_layer_input = np.dot(hidden_layer_2_output, self.weights_hidden_2_output) + self.bias_output
        output_layer_output = self.bipolar_sigmoid(output_layer_input)

        return output_layer_output




data = load_data('seets_dataset.txt')
train_data, test_data = split_data(data)  
nm = NeuralNetwork(train_data, test_data)
nm.print_parameters()


test_instance_index = 0
test_instance = test_data[test_instance_index, :-1]
expected_output = test_data[test_instance_index, -1]

print(f"\nInitial value of the test instance: {test_instance}")
output = nm.feedforward(test_instance)

predicted_label = np.argmax(output) + 1

print(f"Probabilities associated with feedforward: {output}")
print(f"Predicted label: {predicted_label}")
print(f"Expected label: {expected_output}")

if predicted_label == expected_output:
    print("Prediction is correct!")
else:
    print("Prediction is incorrect.")

