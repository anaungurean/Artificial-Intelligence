import numpy as np


def split_data(data):
    unique_labels = np.unique(data[:, -1])
    ok = False
    train_data = None
    test_data = None
    while not ok:
        np.random.shuffle(data)
        split_index = int(0.8 * len(data))
        train_data = data[:split_index] #80% din date sunt date de atrenare
        test_data = data[split_index:] #20% din date sunt date de test
        train_labels = np.unique(train_data[:, -1]) #ne asiguram ca in setul de antrenare avem macar o instanta cu fiecare eticheta (1,2,3)
        #train_data[:, -1] - ia doar ultima coloana
        if all(label in train_labels for label in train_labels):
            ok = True
    return train_data, test_data


def load_data(file_path):
    data = np.loadtxt(file_path, delimiter='\t')
    return data


class NeuralNetwork:

    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

        #Parametrii
        self.input_layer_size = len(self.train_data[0]) - 1 #dimensiunea stratului de intrare
        # dimenstiunea straturilor ascunse
        self.hidden_layer_1_size = 5
        self.hidden_layer_2_size = 4
        self.hidden_layer_3_size = 3
        # dimenstiunea stratului de iesire care e egal cu nr de etichete unice
        self.output_layer_size = len(np.unique(self.train_data[:, -1]))

        self.learning_rate = 0.1 # o folosim la Backpropagation
        self.max_epochs = 500

        self.weights_input_hidden_1 = None #o lista de weights pt fiecere conexiune dintre stratil de intare si primul strat ascuns
        self.weights_hidden_1_hidden_2 = None
        self.weights_hidden_2_hidden_3 = None
        self.weights_hidden_3_output = None






    def print_data(self):
        print(f'Train data is: \n{self.train_data}')
        print(f'Number of train instances is: {len(self.train_data)}\n')
        print(f'Testing data is: \n{self.test_data}')
        print(f'Number of test instances is: {len(self.test_data)}\n')


#Pregatim datele pentru clasificatorul nostru
data = load_data('seets_dataset.txt')
train_data, test_data = split_data(data) #subpunctul 1
nm = NeuralNetwork(train_data, test_data)
nm.print_data()