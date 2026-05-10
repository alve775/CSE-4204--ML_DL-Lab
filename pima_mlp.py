import numpy as np
import pandas as pd


def load_pima_dataset(filename):
    dataframe = pd.read_csv(filename)

    X = dataframe.drop(columns=["Outcome"]).to_numpy(dtype=float)
    y = dataframe["Outcome"].to_numpy(dtype=float).reshape(-1, 1)

    feature_min = X.min(axis=0)
    feature_max = X.max(axis=0)
    feature_range = np.where(feature_max - feature_min == 0, 1, feature_max - feature_min)
    X = (X - feature_min) / feature_range

    return X, y


def train_test_split(X, y, train_ratio=0.7, seed=None):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    train_count = round(train_ratio * len(X))

    train_indices = indices[:train_count]
    test_indices = indices[train_count:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


class BackPropagationNetwork:
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        learning_rate=0.5,
        error_limit=0.01,
        max_iterations=100000,
        seed=None,
    ):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.error_limit = error_limit
        self.max_iterations = max_iterations

        rng = np.random.default_rng(seed)
        self.hidden_weights = rng.uniform(-1.0, 1.0, (input_size, hidden_size))
        self.output_weights = rng.uniform(-1.0, 1.0, (hidden_size, output_size))

        self.hidden_thresholds = rng.uniform(-1.0, 1.0, (1, hidden_size))
        self.output_thresholds = rng.uniform(-1.0, 1.0, (1, output_size))

        self.iterations = 0
        self.error_history = []

    def threshold_function(self, net_input):
        return 1.0 / (1.0 + np.exp(-net_input))

    def threshold_derivative(self, output):
        return output * (1.0 - output)

    def forward(self, X):
        self.hidden_net = np.dot(X, self.hidden_weights) - self.hidden_thresholds
        self.hidden_output = self.threshold_function(self.hidden_net)

        self.output_net = (
            np.dot(self.hidden_output, self.output_weights) - self.output_thresholds
        )
        self.final_output = self.threshold_function(self.output_net)
        return self.final_output

    def total_error(self, y_true, y_pred):
        return 0.5 * np.mean((y_true - y_pred) ** 2)

    def backward(self, X, y_true):
        n_samples = X.shape[0]

        output_error = y_true - self.final_output
        output_delta = output_error * self.threshold_derivative(self.final_output)

        hidden_error = np.dot(output_delta, self.output_weights.T)
        hidden_delta = hidden_error * self.threshold_derivative(self.hidden_output)

        self.output_weights += (
            self.learning_rate
            * np.dot(self.hidden_output.T, output_delta)
            / n_samples
        )
        self.hidden_weights += (
            self.learning_rate * np.dot(X.T, hidden_delta) / n_samples
        )

        self.output_thresholds -= (
            self.learning_rate * np.mean(output_delta, axis=0, keepdims=True)
        )
        self.hidden_thresholds -= (
            self.learning_rate * np.mean(hidden_delta, axis=0, keepdims=True)
        )

    def fit(self, X, y, print_every=1000):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.error_history = []
        self.iterations = 0

        error = float("inf")
        while error > self.error_limit and self.iterations < self.max_iterations:
            y_pred = self.forward(X)
            error = self.total_error(y, y_pred)
            self.error_history.append(error)
            self.backward(X, y)
            self.iterations += 1

            if print_every and (
                self.iterations == 1 or self.iterations % print_every == 0
            ):
                print(f"Iteration {self.iterations:6d} | Error: {error:.8f}")

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return self.forward(X)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)


MLP = BackPropagationNetwork


if __name__ == "__main__":
    X, y = load_pima_dataset("pima-indians-diabetes_original.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_ratio=0.7, seed=7)

    model = BackPropagationNetwork(
        input_size=X.shape[1],
        hidden_size=8,
        output_size=1,
        learning_rate=0.7,
        error_limit=0.03,
        max_iterations=50000,
        seed=7,
    )

    print("Training data:")
    for inputs, target in zip(X_train, y_train.astype(int)):
        print(f"{np.round(inputs, 4).tolist()} -> {target[0]}")

    print("\nTesting data:")
    for inputs, target in zip(X_test, y_test.astype(int)):
        print(f"{np.round(inputs, 4).tolist()} -> {target[0]}")

    model.fit(X_train, y_train, print_every=1000)

    train_predictions = model.predict(X_train)
    test_probabilities = model.predict_proba(X_test)
    test_predictions = model.predict(X_test)
    test_accuracy = np.mean(test_predictions == y_test.astype(int)) * 100

    print("\nRandom hidden thresholds:")
    print(np.round(model.hidden_thresholds, 4))
    print("Random output thresholds:")
    print(np.round(model.output_thresholds, 4))
    print(f"\nTraining stopped after {model.iterations} iterations")
    print(f"Final error: {model.error_history[-1]:.8f}")

    print("\nTraining predictions:")
    for inputs, target, prediction in zip(
        X_train, y_train.astype(int), train_predictions
    ):
        print(
            f"{np.round(inputs, 4).tolist()} -> target={target[0]}, class={prediction[0]}"
        )

    print("\nTesting predictions:")
    for inputs, target, probability, prediction in zip(
        X_test, y_test.astype(int), test_probabilities, test_predictions
    ):
        print(
            f"{np.round(inputs, 4).tolist()} -> target={target[0]}, "
            f"probability={probability[0]:.4f}, "
            f"class={prediction[0]}"
        )
    print(f"Test accuracy: {test_accuracy:.2f}%")
