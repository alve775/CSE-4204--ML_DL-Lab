# Machine Learning Experiments

This repository contains various Machine Learning experiments, implementations, and notebooks, covering algorithms from scratch to advanced deep learning models.

## Repository Contents

*   **Jupyter Notebooks (`*.ipynb`)**:
    *   `deep_learning_book.ipynb`: Notebook containing exercises and code related to deep learning, including Multi-Layer Perceptrons (MLPs) using TensorFlow/Keras and scikit-learn wrappers for cross-validation and grid search.
    *   `mlp.ipynb`, `pima_mlp.ipynb`: Notebooks specifically focusing on Multi-Layer Perceptron implementations and experiments, particularly on the Pima Indians Diabetes dataset.
    *   `work.ipynb`, `work11.ipynb`, `chapter 20.ipynb`, `chapter 20_1.ipynb`, `chapter19.ipynb`: Other exploratory and exercise notebooks.
*   **Python Scripts (`*.py`)**:
    *   `perceptron2.py`, `perceptron3.py`, `percepton.py`, `percepton1.py`: From-scratch implementations of the Perceptron learning algorithm.
    *   `mlp.py`, `pima_mlp.py`: Scripts related to Multi-Layer Perceptron implementations.
    *   `dijkstra.py`, `dijkstra1.py`, `bfs.py`: Graph algorithm implementations.
    *   `bcn.py`: Other utility or algorithm script.
*   **Datasets (`*.csv`)**:
    *   `pima-indians-diabetes_original.csv`: The Pima Indians Diabetes dataset used for binary classification tasks.
    *   `housing.csv`, `iris.csv`: Standard machine learning datasets.
*   **Saved Models (`*.sav`)**:
    *   `finalized_model.sav`: A saved trained model.

## Setup & Requirements

1.  **Environment Setup**:
    It is recommended to use a virtual environment. You can install dependencies using `pip` (or `uv` if preferred, given the presence of `uv.lock`).

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On macOS/Linux
    ```

2.  **Key Dependencies**:
    *   `tensorflow` / `keras`
    *   `scikit-learn`
    *   `scikeras`
    *   `pandas`
    *   `numpy`

    *(Note: For compatibility with scikit-learn 1.6+ and scikeras 0.13, specific patches might be required in the notebooks for `KerasClassifier` usage.)*

## Highlights

### Perceptron Learning Algorithm (From Scratch)

Found in `perceptron2.py` and others, this includes a from-scratch implementation based on textbook steps:
1. Initialize small random weights and threshold.
2. Present input and desired output.
3. Compute actual output with a step function.
4. Adapt weights using basic update, gain rule, or Widrow-Hoff delta rule.

```bash
python3 perceptron2.py
```

### Back Propagation From Scratch

`mlp.py` implements a multilayer back propagation network featuring random weights/thresholds, sigmoid activation, and training stopped by total error.

## Notes

*   Some notebooks and scripts are designed for use in Google Colab or local Jupyter environments.
*   Ensure the datasets (`.csv` files) are in the same directory as the notebooks/scripts when running them.
