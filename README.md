# 🛌 Sleep Efficiency Classification & Regression

This project analyzes sleep health data to **predict and classify sleep efficiency** using modular Python code and machine learning.  
All logic is implemented in `.py` files under the `src/` directory – no Jupyter notebooks needed.

---

## 📊 Project Objectives

- **Regression**: Predict the actual sleep efficiency score (continuous)
- **Classification**: Determine whether sleep efficiency is **sufficient (≥ 0.85)**

---

## 📁 Project Structure

```
datascience-termproject/
├── main.py                         # Main script to run everything
├── src/
│   ├── preprocess.py               # Preprocessing logic
│   ├── scaler_experiment.py        # Compare scalers: Standard, MinMax, Robust
│   ├── model_classification.py     # Classification training, tuning, evaluation
│   └── model_regression.py         # Regression training, tuning, evaluation
├── requirements.txt                # Required Python packages
├── README.md                       # This file
└── .gitignore
```

---

## ⚙️ Features

### 🧹 Preprocessing (`preprocess.py`)
- Extract `hour` from `Bedtime`, `Wakeup time`
- Drop multicollinear columns (e.g., `Sleep duration`)
- Binary encoding of `Gender`, `Smoking status`
- Remove caffeine outliers (IQR method)
- Fill missing values (median/mode by gender)
- Optionally drop `REM sleep %`, `Light sleep %`

### 📈 Scaler Experiment (`scaler_experiment.py`)
- Compares:
  - `StandardScaler`
  - `MinMaxScaler`
  - `RobustScaler`
- Measures RMSE, R² (regression) and accuracy (classification)

### 🔠 Classification (`model_classification.py`)
- Models: Logistic Regression, Decision Tree, Random Forest
- Validation: Stratified K-Fold (k = 3, 5, 10)
- Tuning: `GridSearchCV`, `RandomizedSearchCV`

### 🔢 Regression (`model_regression.py`)
- Models: Linear Regression, Random Forest, Gradient Boosting
- Validation: K-Fold (k = 3, 5, 10)
- Tuning: `GridSearchCV`, `RandomizedSearchCV`

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/datascience-termproject.git
cd datascience-termproject
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Place your dataset

Make sure your data file is in the root directory:

```
ModelingSet.csv
```

### 4. Run the project

```bash
python main.py
```

---

## 🧪 Output

- Scaler comparison results printed to terminal
- Classification model performance summary
- Regression model performance summary

---

## 📦 Requirements

Listed in `requirements.txt`. To install:

```bash
pip install -r requirements.txt
```

Contents:

```
pandas
numpy
scikit-learn
scipy
```

---

## 📜 License

This project is licensed under the **Apache License 2.0**.  
You are free to use, modify, and distribute this code for commercial or non-commercial purposes.  
See the [LICENSE](./LICENSE) file for details.

