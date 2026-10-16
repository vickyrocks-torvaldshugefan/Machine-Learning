# 📡 Telco Customer Churn Prediction

> Predict customer churn using machine learning classification algorithms.

A complete machine learning classification project that predicts whether a telecommunications customer is likely to leave the company using customer demographics, subscribed services, contract details, and billing information.

---

## 🎯 Project Objectives

- 📊 **Explore and preprocess** customer churn data.
- 🧹 **Handle missing values** and prepare categorical and numerical features.
- 🤖 **Build a Logistic Regression** classification model from scratch.
- 📈 **Evaluate model performance** using comprehensive classification metrics.
- 🔍 **Apply cross-validation** to compare different regularization strengths.
- 💡 **Understand business impact** and how machine learning helps identify customers at risk of leaving.

---

## 📂 Dataset Sources

- **Kaggle:** [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Repository Dataset:** [📊 Telco Customer Churn Dataset](https://github.com/vickyrocks-torvaldshugefan/Machine-Learning/blob/main/Classification/Telco-Customer-Churn.csv)

### Dataset Description
The dataset contains customer demographics, subscribed services, contract types, payment methods, monthly charges, total charges, tenure, and churn status (`Churn`).

---

## 🛠️ Technologies and Libraries

- **Python** — Core Programming Language
- **NumPy** — Numerical Computations
- **Pandas** — Data Manipulation and Analysis
- **Matplotlib** — Data Visualization
- **Scikit-learn** — Machine Learning and Model Evaluation
- **Jupyter Notebook** — Interactive Development Environment

---

## ⚙️ Project Workflow

### 1. Data Preparation
- Load the dataset using Pandas.
- Standardize column names and categorical values.
- Convert `totalcharges` into a numeric data type, coercing errors.
- Handle missing values by filling them appropriately (e.g., zeros where applicable).
- Encode the target variable `churn` as binary values:
  - `1` — Customer Churned
  - `0` — Customer Did Not Churn

### 2. Dataset Splitting
Split the dataset into three distinct subsets to prevent data leakage and ensure robust evaluation:
- **Training Set:** 60%
- **Validation Set:** 20%
- **Test Set:** 20%
- *Note:* A fixed random state is used across splits to ensure full reproducibility.

### 3. Feature Engineering
- Separate categorical and numerical features.
- Use `DictVectorizer` from Scikit-learn to transform customer records into numerical feature matrices.
- Train a Logistic Regression model using the encoded feature vectors.

### 4. Model Evaluation
Evaluate the model using multiple comprehensive metrics:
- Accuracy
- Precision & Recall
- Confusion Matrix
- True Positive Rate (TPR) & False Positive Rate (FPR)
- ROC Curve & ROC AUC Score
- K-Fold Cross-Validation

### 5. Hyperparameter Tuning
- Implement **5-Fold Cross-Validation** to compare different values of the Logistic Regression regularization parameter ($C$).

---

## 📊 Model Performance

### Validation Set Results

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 80.34% |
| **Precision** | 67.52% |
| **Recall** | 54.40% |
| **ROC AUC** | 84.38% |

* **Final Test Set ROC AUC:** `85.85%`

### Interpretation
- **Accuracy:** The model correctly classified approximately 80% of validation customers at a standard decision threshold of `0.5`.
- **Precision:** Approximately 67.52% of customers predicted by the model to churn actually churned, reducing false alarms.
- **Recall:** The model successfully identified approximately 54.40% of all actual churners in the dataset.
- **ROC AUC:** The model achieved a score of 0.84 on the validation set, demonstrating a strong capability to rank churners higher than non-churners.

---

## 🧠 Key Concepts Learned

- **Binary Classification** fundamentals.
- **Logistic Regression** probability modeling.
- **Feature Engineering** and categorical feature encoding (`DictVectorizer`).
- **Train, Validation, and Test Splitting** strategies.
- **Classification Thresholds** and their operational trade-offs.
- **Confusion Matrix, Precision, and Recall** dynamics.
- **ROC Curves and AUC** for ranking evaluation.
- **K-Fold Cross-Validation** for robust performance estimation.
- **Regularization and Hyperparameter Tuning** (Parameter $C$).

---

## 💡 Key Takeaways

1. **Accuracy Alone Is Not Enough:** Different evaluation metrics reveal hidden nuances in model performance, especially with skewed target distributions.
2. **Precision Measures Reliability:** It tells us how many predicted churners actually churned, which optimizes retention campaign budgets.
3. **Recall Measures Coverage:** It dictates how many actual churners the business successfully flags before they walk away.
4. **ROC AUC Measures Discrimination:** It evaluates overall ranking quality independent of any specific threshold.
5. **Cross-Validation Improves Reliability:** Testing across multiple data splits ensures stability and guards against overfitting to a single train-test split.

---

## 🔮 Future Improvements

- Compare Logistic Regression performance against advanced models like **Decision Trees**, **Random Forests**, and **Gradient Boosting (XGBoost/LightGBM)**.
- Optimize the classification decision threshold dynamically according to business costs and customer retention campaign budgets.
- Analyze model coefficients deeply to understand primary risk factors associated with customer churn.
- Investigate class imbalance techniques (like SMOTE or class weighting) to improve model recall.

---

## 🎓 Project Purpose

This project demonstrates an end-to-end machine learning classification workflow, covering everything from raw data preprocessing and feature engineering to rigorous model evaluation, cross-validation, and hyperparameter optimization.
