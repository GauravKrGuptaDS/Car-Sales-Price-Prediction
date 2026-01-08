# 🚗 Car Sales Price Prediction using Artificial Neural Network (ANN)

This project builds an **end-to-end Machine Learning pipeline** to predict **car purchase prices** using an **Artificial Neural Network (ANN)**. It includes **feature engineering, preprocessing, model training, evaluation, and model persistence**, making it suitable for **real-world deployment**.

---

## 📌 Problem Statement

Car dealerships often need to estimate how much a customer is likely to spend on a car based on demographic and financial attributes. Manually analyzing this data is time-consuming and error-prone.

This project automates the process by predicting the **car purchase amount** using customer information.

---

## 📊 Dataset Description

The dataset contains the following columns:

| Column Name         | Description                   |
| ------------------- | ----------------------------- |
| customer name       | Customer identifier (dropped) |
| customer e-mail     | Customer identifier (dropped) |
| country             | Customer country              |
| gender              | Customer gender               |
| age                 | Customer age                  |
| annual Salary       | Annual income                 |
| credit card debt    | Outstanding credit card debt  |
| net worth           | Total net worth               |
| car purchase amount | **Target variable**           |

---

## ⚙️ Feature Engineering

To improve model performance, several derived features are created:

### 🔢 Numerical Features

* `age_squared`
* `debt_to_income`
* `networth_to_income`
* `disposable_income`
* `salary_x_age`
* `log_salary`
* `log_net_worth`
* `log_debt`

### 🧩 Categorical Features

* `country`
* `gender`
* `age_group` (derived from age bins)

These features help the ANN capture **non-linear patterns** and **financial behavior**.

---

## 🧪 Preprocessing Pipeline

The project uses **Scikit-learn pipelines** to ensure clean and reproducible preprocessing:

* **Numerical features** → `StandardScaler`
* **Categorical features** → `OneHotEncoder`
* Combined using `ColumnTransformer`

This ensures all inputs are **ANN-ready** and prevents **data leakage**.

---

## 🧠 ANN Model Architecture

The Artificial Neural Network is built using **TensorFlow / Keras**.

```text
Input Layer  → Scaled & Encoded Features
Hidden Layer → 64 neurons (ReLU)
Hidden Layer → 32 neurons (ReLU)
Hidden Layer → 16 neurons (ReLU)
Output Layer → 1 neuron (Regression)
```

* Optimizer: **Adam**
* Loss Function: **Mean Squared Error (MSE)**
* Metric: **Mean Absolute Error (MAE)**
* Regularization: **Dropout + EarlyStopping**

---

## 📈 Model Evaluation

The model is evaluated using:

* **MAE (Mean Absolute Error)**
* **R² Score**

Performance is also visualized using **training vs validation loss curves** to detect overfitting.

---

## 💾 Model Saving & Loading

All critical artifacts are saved using **pickle**:

* Trained ANN model
* Preprocessing pipeline (X scalers & encoders)
* Feature metadata
* Target scaler (optional)

This enables **fast inference without retraining**.

---

## 🧩 Project Structure

```text
car-sales-price-prediction/
│
├── data/
│   └── car_purchasing.csv
│
├── model_artifacts/
│   ├── car_sales_price_prediction_ann.pkl
│   ├── x_preprocessor.pkl
│   └── feature_metadata.pkl
│
├── ANN_Car_Sales_Price_Prediction.ipynb
├── requirements.txt
└── README.md
```

---

## 📦 Technologies Used

* Python
* Pandas & NumPy
* Scikit-learn
* TensorFlow / Keras
* Pickle

---

## 🔮 Future Enhancements

* Compare ANN with **XGBoost / RandomForest**
* Hyperparameter tuning
* Streamlit / FastAPI deployment
* Cloud deployment (AWS / GCP)

---

## 👨‍💻 Author

**Gaurav Kumar Gupta**

AI / Machine Learning Engineer

---

## ⭐ If You Like This Project

Give this repository a ⭐ 
