## Plan: Fix Colab Runtime Warnings

We will update the notebook flow to be Colab-friendly and remove the numeric instability causing `RuntimeWarning` in linear models. The plan is to normalize feature handling consistently, tune sensitive estimators with safer defaults, and make data loading compatible with Colab (`/content` or Drive). This keeps your current workflow intact while making execution stable across Colab sessions.

### Steps
1. Identify warning source in `chapter 20_1.ipynb` around `models`, `cross_val_score`, and unscaled `Lasso`/`ElasticNet`.
2. Replace raw model CV loop with a scaling-first `Pipeline` strategy for all scale-sensitive regressors.
3. Add robust hyperparameters for `Lasso` and `ElasticNet` (`alpha`, `max_iter`, `tol`) to prevent overflow/divergence.
4. Refactor dataset loading cell in `chapter 20_1.ipynb` for Colab paths and optional Drive mount flow.
5. Keep tree-based models unscaled, but separate them from linear/SVM models to avoid unnecessary transforms.
6. Add a minimal warning-check cell to confirm no `divide by zero`/`overflow` appears during cross-validation.

### Further Considerations
1. Which Colab data source do you want? Option A: upload to `/content`, Option B: Google Drive mount, Option C: direct URL.
2. Should we apply the same fixes to `chapter 20.ipynb` since it appears to duplicate the same pipeline?
3. Draft ready for your review; once you confirm, I will provide the exact cell-by-cell changes.

