# K-Means Clustering on Wine Data

_**Early project (2024).** I wrote this as one notebook before moving into industry, and I've kept it as-is to show my starting point. I now structure projects as modular, reusable Python code. See [my current project](https://github.com/jadonc121/causal-inference) for what that looks like._

Unsupervised learning walkthrough: how many groups of wine are hiding in 13 chemical measurements, and how do you justify the answer?

**Notebook:** [K Means Clustering on Wine Data.ipynb](K%20Means%20Clustering%20on%20Wine%20Data.ipynb)

## Problem
Most real-world data has no labels. Clustering finds structure without them, with uses such as customer segmentation and recommendation systems, but it forces you to choose the number of clusters (k) and defend that choice.

## Data
178 wines, 13 continuous features: chemical composition (e.g., malic acid, total phenols, magnesium, proline) and spectral measures (color intensity, hue, OD280). No labels are used, and there are no missing values.

## Approach
- **EDA:** summary statistics before and after standard scaling, correlation heatmap, and scatter matrix with KDE diagonals. Most features are weakly correlated and non-elliptical, which supports using k-means; total phenols and flavanoids are the strongest pair.
- **Dimensionality check:** PCA needed 10 of 13 components to keep 95% of the variance, so no reduction was applied before fitting.
- **Choosing k (k = 1 to 10):**
  - elbow plot of inertia
  - silhouette score for each k
  - silhouette diagrams to check cluster size balance and misassigned points
- **Final model:** k-means with k = 3 on standardized features.
- **Interpretation:** parallel-coordinates plots per cluster show which features distinguish the groups (e.g., phenols, flavanoids, and proanthocyanins are high in one cluster and low in another; color intensity and nonflavanoid phenols stand out in a third).
- **Robustness check:** the model was fit on the full data, then clusters were visualized after compressing to 2 dimensions with PCA, MDS, and LLE.

## Results
- The elbow and the silhouette analysis both pointed to **k = 3** (k = 2 was a close second); silhouette diagrams were acceptable for k = 2 and 3 and showed problems (short or uneven clusters, negative coefficients) for k > 3.
- In 2D projections, **PCA and MDS showed three clear clusters** for k = 3 (and k = 2), even though PCA kept only 55% of the variance. **LLE did not**, and k = 4 and 5 broke down in all three, which supports k = 3.

## Limitations
- There is no external check on the clusters (no comparison to known labels or to alternative models), so "k = 3" is supported by internal metrics only.
- Only k-means was tried; a few features are elliptical, where Gaussian mixture models or DBSCAN could do better.
- Small dataset (178 rows).

## Skills demonstrated
Unsupervised learning, k-means, choosing k (inertia/elbow, silhouette analysis), feature scaling, PCA/MDS/LLE dimensionality reduction, EDA, visualization (matplotlib, seaborn), scikit-learn.

## How to run
Python 3 with `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`. Place `wine-clustering.csv` next to the notebook and run top to bottom.

## Next steps / what I'd do differently today
Compare against Gaussian mixture models and DBSCAN, validate the clusters against the known cultivar labels (adjusted Rand index) if available, and package the workflow as a reusable function or pipeline in modular Python files.
