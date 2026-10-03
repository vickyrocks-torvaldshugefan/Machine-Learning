# 🤖 Machine Learning — Classification

A collection of Machine Learning classification projects built using
Python, Pandas, NumPy, Matplotlib, Seaborn and Scikit-learn.

---

## 📂 Classification Projects

### 1️⃣ Telco Customer Churn Prediction

Predict whether a telecom customer is likely to churn.

### Dataset Sources

- Kaggle: [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)

**Dataset:**  
[📊 Telco Customer Churn Dataset](https://github.com/vickyrocks-torvaldshugefan/Machine-Learning/blob/main/Classification/Telco-Customer-Churn.csv)

**Techniques:**
- Exploratory Data Analysis
- Churn Rate Analysis
- Risk Ratio
- Mutual Information
- Correlation Analysis
- One-Hot Encoding
- DictVectorizer
- Logistic Regression
- Probability Prediction
- Customer-level Prediction
- Model Evaluation

**Model:** Logistic Regression

**Validation Accuracy:** ~80.34%

**Test Accuracy:** ~81.41%

---

### 2️⃣ 💳 Credit Card Default Prediction

Predict whether a credit card client will default on their payment
in the following month.

**Dataset:**  
[📊 Credit Card Dataset](https://github.com/vickyrocks-torvaldshugefan/Machine-Learning/blob/main/Classification/default%20of%20credit%20card%20clients.xls)

**Original Dataset Source:**  
[🌐 UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

**Dataset Information:**
- 30,000 customer records
- 23 predictive features
- Binary classification problem
- Target: `default_payment_next_month`
- `0` → No Default
- `1` → Default

**Features include:**
- Credit limit
- Sex
- Education
- Marriage
- Age
- Repayment history
- Bill statement amounts
- Previous payment amounts

**Techniques:**
- Data Cleaning
- Exploratory Data Analysis
- Feature Importance
- Risk Analysis
- Mutual Information
- Correlation Analysis
- Train / Validation / Test Split
- DictVectorizer
- Logistic Regression
- Probability Prediction
- Individual Customer Prediction
- Model Testing
- Probability Distribution Visualization

**Model:** Logistic Regression

**Validation Accuracy:** ~77.98%

---

## 🧠 Machine Learning Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Analysis
   ↓
Train / Validation / Test Split
   ↓
Feature Transformation
   ↓
Logistic Regression
   ↓
Probability Prediction
   ↓
Classification
   ↓
Model Evaluation
   ↓
Visualization
