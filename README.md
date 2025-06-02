# 🛌 Sleep Efficiency Classification & Regression

This project analyzes sleep health data to **predict and classify sleep efficiency**, using machine learning techniques. It includes full preprocessing, feature engineering, model training, evaluation, and hyperparameter tuning.

---

## 📊 Project Objectives

- **Regression**: Predict the actual sleep efficiency score (continuous value)
- **Classification**: Classify whether a person's sleep efficiency is **"sufficient" (≥ 0.85)** or **"insufficient" (< 0.85)**

---

## 🧹 Preprocessing Steps

All preprocessing is implemented in `src/preprocess.py`.

- **Time Feature Engineering**  
  - `Bedtime`, `Wakeup time` → Extract `hour` features
- **Multicollinearity Removal**  
  - Remove `Sleep duration` and `Deep sleep percentage`
- **Binary Encoding**  
  - Encode categorical variables: `Gender`, `Smoking status`
- **Outlier Elimination**  
  - Remove outliers in `Caffeine consumption` using IQR
- **Missing Value Imputation**  
  - Fill missing values using **median (numeric)** or **mode (categorical)**, stratified by `Gender`
- **Optional Feature Drop**  
  - Remove `REM sleep percentage`, `Light sleep percentage` when specified

---

## ⚙️ Model Overview

### 🔠 Classification

- **Target**: Sleep_Label (1 if efficiency ≥ 0.85, else 0)
- **Models Used**:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
- **Evaluation Metrics**:
  - Accuracy
  - Mean Squared Error (MSE)
  - R² Score
- **Validation**:
  - Stratified K-Fold Cross Validation (k = 3, 5, 10)
- **Tuning**:
  - Performed using `GridSearchCV` and `RandomizedSearchCV` (Random Forest)

📌 **Best Model**:  
✅ Random Forest (RandomSearch, 5-Fold)

---

### 📈 Regression

- **Target**: Sleep efficiency (float)
- **Models Used**:
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
- **Evaluation Metrics**:
  - R² Score
  - Mean Squared Error (MSE)
- **Validation**:
  - K-Fold Cross Validation (k = 3, 5, 10)
- **Tuning**:
  - Random Forest tuned via `GridSearchCV` and `RandomizedSearchCV`

📌 **Best Model**:  
✅ Random Forest Regressor (RandomSearch, 5-Fold)

---

## ⚖️ Scaler Comparison

To evaluate the effect of feature scaling, we compared:

- `StandardScaler`
- `MinMaxScaler`
- `RobustScaler`

**Result:**  
All three scalers produced similar results.  
➡️ `StandardScaler` was chosen for final use due to simplicity and interpretability.

---

## 📁 Repository Structure

```
datascience-termproject/
├── notebooks/
│   └── main.ipynb          # Kaggle notebook with full analysis
├── src/
│   └── preprocess.py       # Preprocessing function
├── LICENSE                 # Apache License 2.0
├── README.md               # This file
├── requirements.txt        # Required packages
└── .gitignore              # Files to ignore in Git
```

---

## 🛠️ How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/datascience-termproject.git
   cd datascience-termproject
   ```

2. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:

   - Open `notebooks/main.ipynb` in Jupyter Notebook or VS Code
   - Execute each cell step by step

4. (Optional) Use `src/preprocess.py` as a reusable preprocessing module for new datasets

---

## 📜 License

This project is licensed under the **Apache License 2.0**.  
You are free to use, modify, and distribute this code for commercial and non-commercial purposes.  
Just include the LICENSE file and indicate any changes you made.

See the [LICENSE](./LICENSE) file for full license text.

