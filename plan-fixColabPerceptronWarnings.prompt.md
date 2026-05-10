## Plan: Fix Colab Warnings and Perceptron Script

Stabilize the Colab workflow by targeting warning-producing training cells and replacing unsafe numerical flow with guarded preprocessing and scale-first modeling. In parallel, provide a clean from-scratch perceptron implementation aligned with textbook update rules (basic, gain-modified, and Widrow-Hoff delta rule).

### Steps
1. Locate warning-producing training code in `work11.ipynb` around `rfe.fit(X, Y)` and `LogisticRegression` calls.
2. Add Colab-safe dataset path handling and explicit numeric coercion.
3. Add `NaN`/`inf` checks before fitting any model.
4. Use `StandardScaler` + `LogisticRegression` pipeline to reduce overflow/divide warnings.
5. Implement textbook perceptron from scratch in `perceptron2.py` with bias `w0 = -theta` and `x0 = 1`.
6. Include basic perceptron, gain-modified update, and Widrow-Hoff delta rule in one runnable script.

### Further Considerations
1. Keep targets binary (`0/1`) for strict textbook parity, or also support bipolar (`-1/+1`).
2. Mirror notebook fixes in other notebooks if they share the same model code.
3. Add a small reproducible demo and seed control for consistent Colab output.

