X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

y = [0, 1, 1, 1]

weights = [0, 0]
bias = 0
lr = 0.1
epochs = 10

def step(value):
    if value >= 0:
        return 1
    return 0

for epoch in range(epochs):
    for i in range(len(X)):
        # calculate weighted sum
        z = 0
        for j in range(len(X[i])):
            z += X[i][j] * weights[j]
        z += bias

        pred = step(z)
        error = y[i] - pred

        # update weights
        for j in range(len(weights)):
            weights[j] = weights[j] + lr * error * X[i][j]

        # update bias
        bias = bias + lr * error

print("Final weights:", weights)
print("Final bias:", bias)

print("Predictions:")
for i in range(len(X)):
    z = 0
    for j in range(len(X[i])):
        z += X[i][j] * weights[j]
    z += bias

    pred = step(z)
    print(X[i], "->", pred)