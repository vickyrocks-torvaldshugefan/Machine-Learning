from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)

from lightgbm import LGBMClassifier


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DATA_PATH = PROJECT_DIR / "online_shoppers_intention.csv"

MODEL_PATH = PROJECT_DIR / "model.pkl"


# --------------------------------------------------
# 2. Load and prepare data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

# Normalize column names
df.columns = df.columns.str.lower().str.replace(" ", "_")

# Normalize categorical text values
categorical_text_columns = list(
    df.dtypes[df.dtypes == "object"].index
)

for col in categorical_text_columns:
    df[col] = (
        df[col]
        .str.lower()
        .str.replace(" ", "_")
    )


# --------------------------------------------------
# 3. Define features and target
# --------------------------------------------------

categorical = [
    "month",
    "operatingsystems",
    "browser",
    "region",
    "traffictype",
    "visitortype",
    "weekend"
]

numerical = [
    "administrative",
    "administrative_duration",
    "informational",
    "informational_duration",
    "productrelated",
    "productrelated_duration",
    "bouncerates",
    "exitrates",
    "pagevalues",
    "specialday"
]

target = "revenue"

X = df[categorical + numerical]
y = df[target].astype(int)


# --------------------------------------------------
# 4. Train / validation / test split
# --------------------------------------------------

df_full_train, df_test, y_full_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1,
    stratify=y
)


# --------------------------------------------------
# 5. Prepare categorical features
# --------------------------------------------------

df_full_train = df_full_train.copy()
df_test = df_test.copy()

for col in categorical:

    df_full_train[col] = (
        df_full_train[col].astype("category")
    )

    df_test[col] = pd.Categorical(
        df_test[col],
        categories=df_full_train[col].cat.categories
    )


# --------------------------------------------------
# 6. Train final LightGBM model
# --------------------------------------------------

best_lgb_params = {
    "n_estimators": 50,
    "learning_rate": 0.05,
    "max_depth": 5,
    "num_leaves": 31,
    "min_child_samples": 10,
    "random_state": 1,
    "n_jobs": -1,
    "verbosity": -1
}

final_model = LGBMClassifier(**best_lgb_params)

final_model.fit(
    df_full_train,
    y_full_train
)


# --------------------------------------------------
# 7. Evaluate on test data
# --------------------------------------------------

y_pred = final_model.predict(df_test)

y_pred_proba = final_model.predict_proba(df_test)[:, 1]

test_auc = roc_auc_score(y_test, y_pred_proba)
test_accuracy = accuracy_score(y_test, y_pred)

print("Test AUC:", round(test_auc, 4))
print("Test Accuracy:", round(test_accuracy, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --------------------------------------------------
# 8. Save the trained model
# --------------------------------------------------

model_artifact = {
    "model": final_model,
    "categorical_features": categorical,
    "numerical_features": numerical,
    "categorical_categories": {
        col: list(df_full_train[col].cat.categories)
        for col in categorical
    }
}

joblib.dump(model_artifact, MODEL_PATH)

print(f"\nModel saved successfully: {MODEL_PATH}")