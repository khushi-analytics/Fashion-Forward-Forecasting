# StyleSense: Fashion Forward Forecasting

A machine learning pipeline that predicts whether a customer would recommend a women's clothing product based on their review text, age, and product category. Built for StyleSense, a fast-growing online retailer, to automatically fill in missing recommendation labels from customer reviews.

The pipeline handles numerical, categorical, and text data — including advanced NLP features such as sentiment polarity and POS-based adjective counts — in a single end-to-end scikit-learn `Pipeline`.

---

## Getting Started

Instructions for how to get a copy of the project running on your local machine.

### Dependencies

```
Python 3.13
pandas
numpy
scikit-learn
nltk
textblob
matplotlib
seaborn
joblib
```

### Installation

Step by step explanation of how to get a dev environment running.

1. **Clone or download the repository**

```bash
git clone https://github.com/khushi-analytics/Fashion-Forward-Forecasting.git
```
2. **Create and activate a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```
3. **Install dependencies using the notebook's install cell**

Run the first cell in `pipeline_project.ipynb`:

```python
import sys
!{sys.executable} -m pip install matplotlib seaborn scikit-learn nltk pandas numpy textblob joblib -q
```

## Testing

Run all cells in `pipeline_project.ipynb` from top to bottom.

### Break Down Tests

**Data loading & missing value check**
Verifies the dataset loads correctly and reports any missing values. Expects 18,442 rows and 9 columns with no nulls.

```python
df.isnull().sum()
```

**Feature engineering check**
Confirms that `Full Review`, `Sentiment Polarity`, and `Adjective Count` columns are created and the raw text columns are dropped.

```python
print("Columns:", df.columns.tolist())
```

**Train/test split**
Verifies that the split is stratified and preserves the ~82/18 class ratio in both sets.

```python
print(y_train.value_counts(normalize=True))
```

**Pipeline training**
Confirms the pipeline fits without errors on training data only.

```python
pipeline.fit(X_train, y_train)
```

**Model evaluation**
Prints accuracy, ROC-AUC, and a full classification report, and displays a confusion matrix and ROC curve.

```python
baseline_metrics = evaluate_model(pipeline, X_test, y_test, title="Baseline Logistic Regression")
```

**Grid search tuning**
Runs a 5-fold stratified cross-validated grid search over regularisation strength, solver, and TF-IDF vocabulary size.

```python
grid_search.fit(X_train, y_train)
print("Best params:", grid_search.best_params_)
```

## Project Instructions

This project builds a supervised machine learning pipeline for StyleSense to predict customer product recommendations. The deliverables are:

**`pipeline_project.ipynb`** — Jupyter Notebook containing the full pipeline from data loading through to evaluation and serialisation. The notebook must run end-to-end without errors via Kernel → Restart & Run All.

**`reviews.csv`** — The dataset used to train and evaluate the model (provided separately).

The pipeline covers:
- Exploratory data analysis with visualisations
- NLP preprocessing (stopword removal, stemming, TF-IDF)
- Advanced NLP features: sentiment polarity (TextBlob) and adjective count (NLTK POS tagging)
- Numerical and categorical feature preprocessing
- Logistic Regression classifier with class imbalance handling
- Hyperparameter tuning via GridSearchCV with stratified 5-fold cross-validation
- Evaluation using accuracy, precision, recall, F1, and ROC-AUC

## Built With

* **[scikit-learn](https://scikit-learn.org/)** - Pipeline, ColumnTransformer, TF-IDF vectorisation, Logistic Regression, GridSearchCV
* **[NLTK](https://www.nltk.org/)** - Stopword removal, Porter stemming, POS tagging for adjective count features
* **[TextBlob](https://textblob.readthedocs.io/)** - Sentiment polarity scoring
* **[pandas](https://pandas.pydata.org/)** - Data loading and manipulation
* **[matplotlib](https://matplotlib.org/) / [seaborn](https://seaborn.pydata.org/)** - Data visualisation
* **[joblib](https://joblib.readthedocs.io/)** - Pipeline serialisation

## License

[License](LICENSE.txt)
