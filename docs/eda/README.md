# Titanic EDA Figures

These figures were exported from the executed Titanic exploratory data analysis notebook (`titanic_eda.ipynb`). The plots summarize data quality, relationships with survival, feature engineering, and exploratory five-fold cross-validation. They are intended for inclusion in the main repository README using paths such as `docs/eda/missing_values.png`.

## Dataset quality

- [Numerical descriptive distributions with mean and median](numeric_distributions.png)
- [Missing-value percentages](missing_values.png)
- [Age missingness patterns](missing_patterns.png)
- [Survival target class balance](class_balance.png)

## Survival relationships

- [Survival rates across categorical features](survival_by_feature.png)
- [Sex by passenger class survival heatmap](sex_pclass_heatmap.png)
- [Combined survival-group plot](survival_groups.png)
- [Age, fare and family counts by survival label](age_fare_distributions.png)

## Feature engineering and evaluation

- [Outlier boxplots](outlier_boxplots.png)
- [Family and derived features vs survival](family_survival.png)
- [Numerical and engineered feature correlations](feature_correlation.png)
- [Five-fold feature ablation (logistic regression)](feature_ablation.png)

**Note:** Figures are exploratory. Associations do not establish causation; the feature-ablation plot is for logistic regression and should not be presented as a PyTorch result. Model preprocessing should always be fitted on training data only.
