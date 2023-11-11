import numpy as np


class NeuralNetwork:
    def __init__(self, file_path):
        self.data = self.load_data(file_path)
        self.train_data, self.test_data = self.split_data(self.data)

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

    def print_data(self):
        print(f'Train data is: \n{self.train_data}')
        print(f'Testing data is: \n{self.test_data}')


nm = NeuralNetwork('seets_dataset.txt')
nm.print_data()




