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
        self.input_layer_size = len(self.train_data[0]) - 1 #dimensiunea (nr de noduri) stratului de intrare
        # dimenstiunea straturilor ascunse
        self.hidden_layer_1_size = 5
        self.hidden_layer_2_size = 4
        self.hidden_layer_3_size = 3
        # dimenstiunea stratului de iesire care e egal cu nr de etichete unice
        self.output_layer_size = len(np.unique(self.train_data[:, -1]))

        self.learning_rate = 0.1 # o folosim la Backpropagation
        #epoca = o singură trecere completă prin întregul set de date de antrenament, daca avem prea multe -> overfitting
        self.max_epochs = 500

        #Ponderi
        #determina importanța fiecărei intrări în calculul ieșirii.
        #influențează cât de mult contribuie intrarea unui neuron la activarea celui următor
        self.weights_input_hidden_1 = None #o lista de weights pt fiecere conexiune dintre stratul de intare si primul strat ascuns
        self.weights_hidden_1_hidden_2 = None
        self.weights_hidden_2_hidden_3 = None
        self.weights_hidden_3_output = None
        self.assign_weights()

        #Bias-uri parametru adițional - val constanta
        #Ne asiguram că chiar și când toate intrările sunt zero,
        # neuronul poate avea o ieșire non-nulă.
        #Se adăuga la suma ponderată a intrărilor înainte de a fi transmisă funcției de activare
        self.bias_hidden_1 = None
        self.bias_hidden_2 = None
        self.bias_hidden_3 = None
        self.bias_output = None
        self.assign_biases()

    def assign_weights(self):
        ok = False
        while not ok:
            #asignam o matrice de valori random intre -0.1 si 0.1
            #Fiecare neuron din stratul de intrare este conectat la fiecare neuron din primul strat ascuns.
            #Numărul de ponderi între aceste două straturi este egal cu produsul numărului de neuroni din stratul de intrare și numărului de neuroni din primul strat ascuns.
            self.weights_input_hidden_1 = np.random.uniform(-0.1, 0.1,
                                                            (self.input_layer_size, self.hidden_layer_1_size))
            self.weights_hidden_1_hidden_2 = np.random.uniform(-0.1, 0.1,
                                                               (self.hidden_layer_1_size, self.hidden_layer_2_size))
            self.weights_hidden_2_hidden_3 = np.random.uniform(-0.1, 0.1,
                                                               (self.hidden_layer_2_size, self.hidden_layer_3_size))
            self.weights_hidden_3_output = np.random.uniform(-0.1, 0.1,
                                                             (self.hidden_layer_3_size, self.output_layer_size))
            if not(self.weights_hidden_1_hidden_2.any() == 0 or self.weights_hidden_2_hidden_3.any() == 0 or self.weights_hidden_3_output.any() == 0):
                ok = True

    def assign_biases(self):
        ok = False
        while not ok:
            self.bias_hidden_1 = np.random.uniform(-0.1, 0.1, (1, self.hidden_layer_1_size))
            self.bias_hidden_2 = np.random.uniform(-0.1, 0.1, (1, self.hidden_layer_2_size))
            self.bias_hidden_3 = np.random.uniform(-0.1, 0.1, (1, self.hidden_layer_3_size))
            self.bias_output = np.random.uniform(-0.1, 0.1, (1, self.output_layer_size))
            if not (np.any(self.bias_hidden_1) == 0 or np.any(self.bias_hidden_2) == 0 or
                    np.any(self.bias_hidden_3) == 0 or np.any(self.bias_output) == 0):
                ok = True

    def print_parameters(self):
        print("==== Neural Network Parameters ====")
        print(f"Input Layer Size: {self.input_layer_size}")
        print(f"Hidden Layer 1 Size: {self.hidden_layer_1_size}")
        print(f"Hidden Layer 2 Size: {self.hidden_layer_2_size}")
        print(f"Hidden Layer 3 Size: {self.hidden_layer_3_size}")
        print(f"Output Layer Size: {self.output_layer_size}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Maximum Epochs: {self.max_epochs}")
        print("\n==== Weights ====")
        print(f"Input-Hidden 1 Weights:\n{self.weights_input_hidden_1}")
        print(f"Hidden 1-Hidden 2 Weights:\n{self.weights_hidden_1_hidden_2}")
        print(f"Hidden 2-Hidden 3 Weights:\n{self.weights_hidden_2_hidden_3}")
        print(f"Hidden 3-Output Weights:\n{self.weights_hidden_3_output}")
        print("\n==== Biases ====")
        print(f"Hidden 1 Bias:\n{self.bias_hidden_1}")
        print(f"Hidden 2 Bias:\n{self.bias_hidden_2}")
        print(f"Hidden 3 Bias:\n{self.bias_hidden_3}")
        print(f"Output Bias:\n{self.bias_output}")


    #-----Functiile de activare -----
    # decide cât de mult un neuron artificial trebuie să fie activat, adică cât de mult semnalul procesat de neuron trebuie să influențeze rețeaua mai departe.

    # funcția sigmoid normalizează valorile în intervalul de la 0 la 1.
    # Mai specific, cât de mare este valoarea de intrare (în sens pozitiv), cu atât rezultatul funcției sigmoid se va apropia de 1.
    # cu cât valoarea de intrare este mai mică (intrând în spațiul negativ), cu atât rezultatul funcției se va apropia de 0.
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    #Returnează 0 pentru orice valoare negativă a lui x și valoarea lui x pentru orice valoare pozitivă a lui x.
    def relu(self, x):
        return np.maximum(0, x)

    #Este 0 pentru orice valoare negativă a lui x și 1 pentru orice valoare pozitivă a lui x.
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    #mapează intrările la valori între -1 și 1
    def bipolar_sigmoid(self, x):
        return (1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x))

    def bipolar_sigmoid_derivative(self, x):
        return 1 - self.bipolar_sigmoid(x)**2

    #evalueaza cât de bine modelul efectuează predicții comparativ cu valorile adevărate
    #ofera o măsură cantitativă a discrepanței dintre predicțiile modelului și valorile reale
    def error_function(self, target, output):
        return 0.5 * np.mean((target - output) ** 2)

    def feedforward(self, input_data):
        #np dot efectuează o înmulțire matriceală între intrările (sau ieșirile) de la un strat și ponderile care conectează acel strat la următorul
        hidden_layer_1_input = np.dot(input_data, self.weights_input_hidden_1) + self.bias_hidden_1
        hidden_layer_1_output = self.sigmoid(hidden_layer_1_input)

        hidden_layer_2_input = np.dot(hidden_layer_1_output, self.weights_hidden_1_hidden_2) + self.bias_hidden_2
        hidden_layer_2_output = self.relu(hidden_layer_2_input)

        hidden_layer_3_input = np.dot(hidden_layer_2_output, self.weights_hidden_2_hidden_3) + self.bias_hidden_3
        hidden_layer_3_output = self.bipolar_sigmoid(hidden_layer_3_input)

        output_layer_input = np.dot(hidden_layer_3_output, self.weights_hidden_3_output) + self.bias_output
        output_layer_output = self.sigmoid(output_layer_input)

        return output_layer_output




#Pregatim datele pentru clasificatorul nostru
data = load_data('seets_dataset.txt')
train_data, test_data = split_data(data) #subpunctul 1
nm = NeuralNetwork(train_data, test_data)
nm.print_parameters()

