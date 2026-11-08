# 💳 Credit Scoring and Default Prediction Using Machine Learning

## 📌 Project Overview
This project focuses on **Credit Risk Assessment** and **Default Prediction** using Machine Learning. The main objective is to predict whether a loan applicant is likely to repay their loan or default based on their personal, employment, and financial information.

Credit risk assessment is an important task in the banking and financial industry. Financial institutions need to evaluate applicants before providing loans. Machine Learning can help identify patterns in historical credit data and estimate the likelihood of loan default.

In this project, I explore and implement multiple classification algorithms, including **Decision Trees**, **Random Forest**, and **XGBoost**, to predict credit default and compare their performance.

---

## 🎯 Project Objectives
- Understand and explore the credit scoring dataset.
- Perform data cleaning and preprocessing.
- Handle missing values and unknown records.
- Analyze important features related to credit risk.
- Build multiple Machine Learning classification models.
- Perform hyperparameter tuning to improve model performance.
- Compare models using the **ROC-AUC** score.
- Select the best-performing model based on evaluation results.

---

## 📂 Dataset Information
The dataset used in this project is the **Credit Scoring Dataset**, originally published by Gaston Sanchez.

- **Original Dataset:** [Gaston Sanchez – Credit Scoring](https://github.com/gastonstat/CreditScoring)
- - **Dataset File:** [CreditScoring.csv](https://raw.githubusercontent.com/vickyrocks-torvaldshugefan/Machine-Learning/main/06-Trees/CreditScoring.csv)
- **Format:** CSV
- **Number of Records:** 4,455
- **Number of Features:** 14 (including the target variable)
- **Problem Type:** Binary Classification
- **Target Variable:** `status`

### Target Variable Breakdown
The `status` column represents the credit status of an applicant:

| Original Value | Meaning |
| :--- | :--- |
| `1` | **OK** – Credit repaid / non-default |
| `2` | **Default** – Credit default |
| `0` | **Unknown** – Missing or unknown status |

For this project, the unknown target record ($0$) is removed, and the prediction task is converted into binary classification:
- $0 \rightarrow \text{OK}$
- $1 \rightarrow \text{Default}$

*After removing the unknown target record, the dataset contains **4,454 records**.*

---

## 📊 Dataset Features
The dataset contains information about applicants' personal details, employment, and financial circumstances:

| Feature | Description |
| :--- | :--- |
| `Status` | Credit repayment status (target variable) |
| `Seniority` | Job seniority in years |
| `Home` | Type of home ownership |
| `Time` | Requested loan duration |
| `Age` | Applicant's age |
| `Marital` | Marital status |
| `Records` | Existence of credit records |
| `Job` | Type of employment |
| `Expenses` | Applicant's expenses |
| `Income` | Applicant's income |
| `Assets` | Applicant's assets |
| `Debt` | Applicant's debt |
| `Amount` | Requested loan amount |
| `Price` | Price of the financed goods |

---

## 🛠️ Technologies and Libraries

| Technology / Library | Purpose |
| :--- | :--- |
| **Python** | Programming language |
| **Jupyter Notebook** | Development and experimentation |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computations |
| **Matplotlib** | Data visualization |
| **Seaborn** | Statistical visualization |
| **Scikit-learn** | Machine Learning models and evaluation |
| **XGBoost** | Gradient boosting classification |

---

## 🔍 Project Workflow

### 1. Data Loading and Exploration
- Load the CSV dataset using Pandas.
- Examine dataset structure, dimensions, column names, and data types.
- Explore numerical and categorical variables.
- Analyze the distribution of the target variable.

### 2. Data Cleaning and Preprocessing
- Convert column names to lowercase.
- Map numerical categorical values into meaningful text labels.
- Convert unknown values into appropriate representations.
- Identify missing values and replace placeholder values (e.g., `99999999` in `income`, `assets`, and `debt`) with `NaN`.
- Remove records with unknown target status (`status = 0`).
- Prepare the dataset for Machine Learning.

### 3. Train, Validation, and Test Split
The dataset is split into three subsets:

| Subset | Percentage | Purpose |
| :--- | :--- | :--- |
| **Training** | 60% | Train models |
| **Validation** | 20% | Model selection & hyperparameter tuning |
| **Testing** | 20% | Final model evaluation |

### 4. Decision Tree Classification
- Splitting data based on feature thresholds.
- Exploring tree depth.
- Hyperparameter tuning using `max_depth` and `min_samples_leaf`.
- Evaluating training and validation performance.

### 5. Random Forest Classification
- Training multiple decision trees (Ensemble Learning).
- Experimenting with the number of estimators (`n_estimators`).
- Tuning `max_depth` and `min_samples_leaf`.
- Comparing performance metrics.

### 6. Gradient Boosting and XGBoost
- Understanding gradient boosting principles.
- Training an XGBoost classifier with binary logistic objective (`binary:logistic`).
- Evaluating performance using ROC-AUC.
- Experimenting with boosting iterations (`num_boost_round`).
- Hyperparameter tuning for `eta`, `max_depth`, `min_child_weight`, and `num_boost_round`.

### 7. Model Selection and Evaluation
- Models are compared using the **ROC-AUC** (Area Under the Receiver Operating Characteristic Curve) metric.
- The top-performing model on validation data is re-trained on the full training set and evaluated on the test set.

---

## 📈 Model Performance

### Validation Results

| Model | Validation ROC-AUC |
| :--- | :--- |
| **Decision Tree** | 0.7851 |
| **Random Forest** | 0.8246 |
| **XGBoost** | **0.8351** |

### Final Test Result
The selected **XGBoost** model was retrained using the full training dataset and evaluated on the test dataset.

| Model | Test ROC-AUC |
| :--- | :--- |
| **XGBoost** | **0.8291** |

> **Observation:** XGBoost achieved the highest validation ROC-AUC among the three evaluated models and obtained a test ROC-AUC of approximately **0.8291**. ROC-AUC measures how well a model distinguishes between the two classes across different classification thresholds.

---

## 📁 Repository Structure

```text
Machine-Learning/
│
├── 06-Trees/
│   └── CreditScoring.csv
│
└── README.md
```

---

## 🚀 How to Run This Project

### Step 1: Clone the Repository
```bash
git clone https://github.com/vickyrocks-torvaldshugefan/Machine-Learning.git
```

### Step 2: Navigate to the Project Directory
```bash
cd Machine-Learning
```

### Step 3: Install Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

### Step 4: Open Jupyter Notebook
```bash
jupyter notebook
```

### Step 5: Load and Inspect the Dataset
```python
import pandas as pd

df = pd.read_csv("06-Trees/CreditScoring.csv")
df.head()
```

---

## 📚 Key Learnings
Through this project, I explored and practiced:
- Data cleaning, preprocessing, and categorical feature encoding.
- Handling missing values and placeholders.
- Binary classification techniques.
- Decision Trees and hyperparameter tuning (`max_depth`, `min_samples_leaf`).
- Ensemble methods including **Random Forests** and **Gradient Boosting (XGBoost)**.
- Evaluating classification models using **ROC-AUC**.

---

## ⚠️ Disclaimer
This project is developed for educational and Machine Learning practice purposes. The predictions are not intended to be used as the sole basis for real-world financial or lending decisions.

---

## 🙏 Acknowledgements
- **Original Dataset:** Gaston Sanchez – Credit Scoring
- **Learning Resource:** Machine Learning Zoomcamp

---

⭐ *If you find this project interesting, feel free to star or explore the repository!*
