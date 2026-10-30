# Car Fuel Efficiency Prediction
**Machine Learning Zoomcamp – Homework**

## Objective
Build regression models to predict car fuel efficiency (`fuel_efficiency_mpg`) using the 2026 fuel efficiency dataset.

## Dataset & Preprocessing
* **Dataset:** Car Fuel Efficiency 2026
* **Missing Values:** Handled missing values using `.fillna(0)`.
* **Data Splitting:** Split dataset into training, validation, and test sets.
* **Feature Encoding:** Transformed features using `DictVectorizer(sparse=True)`.

---

## Questions & Results

### Q1. Decision Tree Regressor
* **Model:** `DecisionTreeRegressor`
* **Parameter:** `max_depth=1`
* **Result:** `model_year`

### Q2. Random Forest Regressor
* **Parameters:** `n_estimators=10`, `random_state=1`
* **Validation RMSE:** `1.806789`
* **Closest Option:** `1.837`

### Q3. Tuning `n_estimators`
Tested values: `[10, 50, 100, 150]`

| `n_estimators` | RMSE |
| :--- | :--- |
| **10** | 1.807 |
| **50** | 1.747 |
| **100** | 1.739 |
| **150** | 1.736 |

* **Best Value:** `150`

### Q4. Tuning `max_depth`
Tested values: `[10, 15, 20, 25]`

| `max_depth` | Mean RMSE |
| :--- | :--- |
| **10** | **1.726** |
| **15** | 1.754 |
| **20** | 1.758 |
| **25** | 1.757 |

* **Best Value:** `10`

### Q5. Feature Importance
* **Model Parameters:** `n_estimators=10`, `max_depth=20`, `random_state=1`

| Feature | Importance |
| :--- | :--- |
| `model_year` | 0.305160 |
| `vehicle_weight` | 0.203259 |
| `fuel_type=Gasoline` | 0.144707 |
| `horsepower` | 0.076278 |
| `engine_displacement` | 0.063131 |

* **Most important among given options:** `vehicle_weight`

### Q6. XGBoost – Learning Rate Tuning
Compared two learning rates (`eta`) using 100 boosting rounds:

| `eta` | Validation RMSE |
| :--- | :--- |
| **0.3** | 1.805297 |
| **0.1** | **1.707464** |

* **Best Value:** `0.1`

---

## Final Answers Summary

| Question | Answer |
| :--- | :--- |
| **Q1** | `model_year` |
| **Q2** | `1.837` (closest option) |
| **Q3** | `150` |
| **Q4** | `10` |
| **Q5** | `vehicle_weight` |
| **Q6** | `0.1` |

---

## Key Learnings
* Decision Trees and Random Forests implementation in Python.
* Evaluating regression models using Root Mean Squared Error (RMSE).
* Systematic hyperparameter tuning (`n_estimators`, `max_depth`, `eta`).
* Interpreting feature importance in ensemble models.
* Training and tuning gradient boosting models using XGBoost.

## Conclusion
Successfully built regression models, tuned key hyperparameters, and evaluated comparative performance using validation RMSE to determine optimal model setups.
