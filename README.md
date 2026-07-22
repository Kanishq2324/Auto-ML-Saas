\# AutoML SaaS Engine



A reusable tabular AutoML engine that automatically:



\- Detects classification or regression tasks

\- Cleans tabular datasets

\- Builds preprocessing pipelines

\- Compares multiple machine-learning models

\- Performs cross-validation

\- Tunes the strongest models

\- Evaluates the final model

\- Exports trained pipelines and experiment artifacts



\## Current models



\### Classification



\- Logistic Regression

\- K-Nearest Neighbors

\- Random Forest

\- Extra Trees

\- XGBoost



\### Regression



\- Ridge Regression

\- K-Nearest Neighbors

\- Random Forest

\- Extra Trees

\- XGBoost



\## Example



```python

result = run\_automl(

&#x20;   csv\_path="data/insurance.csv",

&#x20;   target\_column="charges",

&#x20;   task="auto",

&#x20;   tuning\_mode="fast"

)

