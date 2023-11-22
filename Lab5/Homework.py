import NeuronalNetwork
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
def main():
    data = load_data('seets_dataset.txt')
    train_data, test_data = split_data(data)
    nm = NeuronalNetwork.NeuralNetwork(train_data, test_data)
    # nm.print_parameters()

    nm.train()
    nm.test()

if __name__ == '__main__':
    main()