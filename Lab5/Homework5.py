import numpy as np


class NeuralNetwork:
    def __init__(self, file_path):
        self.data = self.load_data(file_path)
        self.train_data, self.test_data = self.split_data(self.data)

        # Parametrii
        self.input_layer_size = self.train_data.shape[1] - 1
        self.hidden_layer_1_size = 5
        self.hidden_layer_2_size = 4
        self.hidden_layer_3_size = 3
        self.output_layer_size = len(np.unique(self.data[:, -1]))

        self.learning_rate = 0.1
        self.max_epochs = 500

        # Ponderile
        self.weights_input_hidden_1 = None
        self.weights_hidden_1_hidden_2 = None
        self.weights_hidden_2_hidden_3 = None
        self.weights_hidden_3_output = None
        self.assign_weights()

        #Bias-uri
        self.bias_hidden_1 = None
        self.bias_hidden_2 = None
        self.bias_hidden_3 = None
        self.bias_output = None
        self.assign_biases()

    def assign_weights(self):
        ok = True
        while ok:
            self.weights_input_hidden_1 = np.random.uniform(-0.1, 0.1,(self.input_layer_size, self.hidden_layer_1_size))
            self.weights_hidden_1_hidden_2 = np.random.uniform(-0.1, 0.1,(self.hidden_layer_1_size, self.hidden_layer_2_size))
            self.weights_hidden_2_hidden_3 = np.random.uniform(-0.1, 0.1,(self.hidden_layer_2_size, self.hidden_layer_3_size))
            self.weights_hidden_3_output = np.random.uniform(-0.1, 0.1,(self.hidden_layer_3_size, self.output_layer_size))
            if not (self.weights_hidden_1_hidden_2.any() == 0 or self.weights_hidden_2_hidden_3.any() == 0 or self.weights_hidden_3_output.any() == 0):
                ok = False

    def assign_biases(self):
        ok = True
        while ok:
            self.bias_hidden_1 = np.random.uniform(-0.1, 0.1, (1, self.hidden_layer_1_size))
            self.bias_hidden_2 = np.random.uniform(-0.1, 0.1, (1, self.hidden_layer_2_size))
            self.bias_hidden_3 = np.random.uniform(-0.1, 0.1, (1, self.hidden_layer_3_size))
            self.bias_output = np.random.uniform(-0.1, 0.1, (1, self.output_layer_size))
            if not (np.any(self.bias_hidden_1) == 0 or np.any(self.bias_hidden_2) == 0 or
                    np.any(self.bias_hidden_3) == 0 or np.any(self.bias_output) == 0):
                ok = False

    def load_data(self, file_path):
        data = np.loadtxt(file_path, delimiter='\t')
        return data

    def split_data(self, data):
        np.random.shuffle(data)
        split_index = int(0.8 * len(data))
        train_data = data[:split_index]
        test_data = data[split_index:]
        return train_data, test_data

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def bipolar_sigmoid(self, x):
        return (1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x))

    def bipolar_sigmoid_derivative(self, x):
        return 1 - x**2

    def error_function(self, target, output):
        return 0.5 * np.mean((target - output) ** 2)

    def print_data(self):
        print(f'Train data is: \n{self.train_data}')
        print(f'Testing data is: \n{self.test_data}')

    def feedforward(self, input_data):
        hidden_layer_1_input = np.dot(input_data, self.weights_input_hidden_1) + self.bias_hidden_1
        hidden_layer_1_output = self.sigmoid(hidden_layer_1_input)

        hidden_layer_2_input = np.dot(hidden_layer_1_output, self.weights_hidden_1_hidden_2) + self.bias_hidden_2
        hidden_layer_2_output = self.relu(hidden_layer_2_input)

        hidden_layer_3_input = np.dot(hidden_layer_2_output, self.weights_hidden_2_hidden_3) + self.bias_hidden_3
        hidden_layer_3_output = self.bipolar_sigmoid(hidden_layer_3_input)

        output_layer_input = np.dot(hidden_layer_3_output, self.weights_hidden_3_output) + self.bias_output
        output_layer_output = self.sigmoid(output_layer_input)

        return output_layer_output, hidden_layer_3_output, hidden_layer_2_output, hidden_layer_1_output

    def backpropagation(self, input_data, target):
        output_layer_output, hidden_layer_3_output, hidden_layer_2_output, hidden_layer_1_output = self.feedforward(input_data)

        output_layer_error = target - output_layer_output
        output_layer_delta = output_layer_error * self.sigmoid_derivative(output_layer_output)

        hidden_layer_3_error = np.dot(output_layer_delta, self.weights_hidden_3_output.T)
        hidden_layer_3_delta = hidden_layer_3_error * self.bipolar_sigmoid_derivative(hidden_layer_3_output)

        hidden_layer_2_error = np.dot(hidden_layer_3_delta, self.weights_hidden_2_hidden_3.T)
        hidden_layer_2_delta = hidden_layer_2_error * self.relu_derivative(hidden_layer_2_output)

        hidden_layer_1_error = np.dot(hidden_layer_2_delta, self.weights_hidden_1_hidden_2.T)
        hidden_layer_1_delta = hidden_layer_1_error * self.sigmoid_derivative(hidden_layer_1_output)

        #actualizam ponderile
        self.weights_hidden_3_output += self.learning_rate * np.dot(hidden_layer_3_output.T.reshape(-1, 1),
                                                                    output_layer_delta.reshape(1, -1))
        self.weights_hidden_2_hidden_3 += self.learning_rate * np.dot(hidden_layer_2_output.T.reshape(-1, 1),
                                                                      hidden_layer_3_delta.reshape(1, -1))
        self.weights_hidden_1_hidden_2 += self.learning_rate * np.dot(hidden_layer_1_output.T.reshape(-1, 1),
                                                                      hidden_layer_2_delta.reshape(1, -1))
        self.weights_input_hidden_1 += self.learning_rate * np.dot(input_data.reshape(-1, 1),
                                                                   hidden_layer_1_delta.reshape(1, -1))

        #actualizam bias-urile
        self.bias_output += self.learning_rate * output_layer_delta
        self.bias_hidden_3 += self.learning_rate * hidden_layer_3_delta
        self.bias_hidden_2 += self.learning_rate * hidden_layer_2_delta
        self.bias_hidden_1 += self.learning_rate * hidden_layer_1_delta
        return output_layer_error

    def train(self):
        total_error = 0
        for epoch in range(self.max_epochs):
            epoch_error = 0
            for input_data in self.train_data:
                target = np.zeros(self.output_layer_size)
                label = int(input_data[-1])

                # Adjust the label if necessary
                label_index = label - 1  # Subtract 1 if your labels start from 1

                target[label_index] = 1
                output_layer_error = self.backpropagation(input_data[:-1], target)
                epoch_error += self.error_function(target, output_layer_error)

            epoch_mean_error = epoch_error / len(self.train_data)
            total_error += epoch_mean_error
            if epoch % 100 == 0:
                print(f'Epoch: {epoch}, error: {epoch_mean_error}')

        average_error = total_error / self.max_epochs
        print(f'Average error over all epochs: {average_error}')

    '''predicția pe setul de date de testare și afișarea metricilor de performanță'''

    def run_test(self):
        predictions = []
        labels = []

        for input_data in self.test_data:
            label = int(input_data[-1])
            labels.append(label)
            prediction = self.predict(input_data[:-1])
            predictions.append(prediction)

        self.evaluate_performance(labels, predictions)

    def predict(self, input_data):
        output_layer_output, _, _, _ = self.feedforward(input_data)
        prediction = np.argmax(output_layer_output) + 1
        return prediction

    def evaluate_performance(self, labels, predictions):
        unique_classes = set(labels)
        conf_matrix = {cls: {cls_pred: 0 for cls_pred in unique_classes} for cls in unique_classes}

        for label, prediction in zip(labels, predictions):
            conf_matrix[label][prediction] += 1

        print("Matricea de confuzie:")
        for cls, predictions in conf_matrix.items():
            print(f"{cls}: {predictions}")

        precision = {}
        recall = {}
        f1_score = {}

        for cls in unique_classes:
            tp = conf_matrix[cls][cls]
            fp = sum(conf_matrix[cls_pred][cls] for cls_pred in unique_classes if cls_pred != cls)
            fn = sum(conf_matrix[cls][cls_pred] for cls_pred in unique_classes if cls_pred != cls)
            precision[cls] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score[cls] = 2 * precision[cls] * recall[cls] / (precision[cls] + recall[cls]) if (precision[cls] +
                                                                                                  recall[
                                                                                                      cls]) > 0 else 0

        weighted_precision = sum(
            precision[cls] * sum(label == cls for label in labels) for cls in unique_classes) / len(labels)
        weighted_recall = sum(recall[cls] * sum(label == cls for label in labels) for cls in unique_classes) / len(
            labels)
        weighted_f1 = sum(f1_score[cls] * sum(label == cls for label in labels) for cls in unique_classes) / len(labels)

        accuracy = sum(l == p for l, p in zip(labels, predictions)) / len(labels)

        print(f"Accuracy: {accuracy}")
        print(f"Weighted Precision: {weighted_precision}")
        print(f"Weighted Recall: {weighted_recall}")
        print(f"Weighted F1 Score: {weighted_f1}")


# Exemplu de utilizare
nm = NeuralNetwork('seets_dataset.txt')
nm.train()
nm.run_test()

