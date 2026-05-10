import random
from re import split


def create_dataset():
    dataset = []
    for i in range(16):
        bits = format(i, "04b")
        x = [int(bit) for bit in bits]
        if i < 8:
            y = 0
            dataset.append((x, y))
        else:
            y = 1
            dataset.append((x, y))

    return dataset


def split_dataset(dataset, split_number):
    random.shuffle(dataset)
    train_count = round((split_number/10)*len(dataset))
    train_data = dataset[:train_count]
    test_data = dataset[train_count:]

    return train_data, test_data



def predict(x, weights):
    net = 0
    for i in range(4):
        net += weights[i] * x[i]
    if net > 0:
        return 1
    else:
        return 0


def train_perceptron(train_data, max_epochs=20):
    weights = []
    for i in range(4):
        weights.append(random.uniform(0, 1))
    print("Initial weights: ",)

    for epoch in range(max_epochs):
        error = 0
        for x, target in train_data:
            output = predict(x, weights)
            if output == target:
                pass
            elif output == 0 and target == 1:
                for i in range(4):
                    weights[i] = weights[i] + x[i]
                error += 1
            elif output == 1 and target == 0:
                for i in range(4):
                        weights[i] = weights[i] - x[i]
                error += 1

        print("Epoch", epoch+1, "weights: ", weights)

    return weights


def test_perceptron(data, weights):
    correct = 0
    for x, target in data:
        output = predict(x, weights)
        print(x, "Target =", target, "Predicted =", output)
        if output == target:
            correct += 1

    if len(data) > 0:
        accuracy = (correct / len(data))*100
        print("Accuracy: ", accuracy, "%")


if __name__ == "__main__":
    split_num = int(input("Split number: "))
    dataset = create_dataset()
    print("Whole dataset: ")
    print(dataset)
    train_data, test_data = split_dataset(dataset, split_num)
    print("Train data: ",train_data)
    print("Test data: ",test_data)

    # for x,y in train_data:
    #     print(x, "->", y)

    weights = train_perceptron(train_data)
    print("Final weights: ", weights)

    print("For test data:")
    # test_perceptron(train_data, weights)
    test_perceptron(test_data, weights)