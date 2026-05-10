import random

# Create dataset from 0000 to 1111
def create_dataset():
    dataset = []

    for i in range(16):
        bits = format(i, "04b")
        x = [int(bit) for bit in bits]

        if i < 8:
            y = 0
        else:
            y = 1

        dataset.append((x, y))

    return dataset


# Split whole dataset, not class-wise
def split_dataset(dataset, split_number):
    random.shuffle(dataset)

    train_count = round((split_number / 10) * len(dataset))

    train_data = dataset[:train_count]
    test_data = dataset[train_count:]

    return train_data, test_data


# Predict output
def predict(x, weights):
    net = 0
    for i in range(4):
        net += x[i] * weights[i]

    if net > 0:
        return 1
    else:
        return 0


# Train perceptron using your update rule
def train_perceptron(train_data, max_epochs=20):
    weights = [random.uniform(0, 1) for _ in range(4)]

    print("Initial weights:", [round(w, 3) for w in weights])

    for epoch in range(max_epochs):
        errors = 0

        for x, target in train_data:
            output = predict(x, weights)

            if output == target:
                pass
            elif output == 0 and target == 1:
                for i in range(4):
                    weights[i] = weights[i] + x[i]
                errors += 1
            elif output == 1 and target == 0:
                for i in range(4):
                    weights[i] = weights[i] - x[i]
                errors += 1

        print("Epoch", epoch + 1, "weights:", [round(w, 3) for w in weights])

        # if errors == 0:
        #     print("Training finished early")
        #     break

    return weights


# Test model
def test_perceptron(data, weights, title):
    print("\n" + title)
    correct = 0

    for x, target in data:
        output = predict(x, weights)
        print(x, "Target =", target, "Predicted =", output)

        if output == target:
            correct += 1

    if len(data) > 0:
        accuracy = (correct / len(data)) * 100
        print("Accuracy =", round(accuracy, 2), "%")
    else:
        print("No data in this set")


# Main
split_number = int(input("Enter a number from 1 to 10: "))

if split_number < 1 or split_number > 10:
    print("Invalid input")
else:
    dataset = create_dataset()
    train_data, test_data = split_dataset(dataset, split_number)

    print("\nTraining data:")
    for x, y in train_data:
        print(x, "->", y)

    print("\nTesting data:")
    for x, y in test_data:
        print(x, "->", y)

    weights = train_perceptron(train_data)

    print("\nFinal weights:", [round(w, 3) for w in weights])

    test_perceptron(train_data, weights, "Training Results")
    test_perceptron(test_data, weights, "Testing Results")