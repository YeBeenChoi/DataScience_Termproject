# 🛌 Sleep Efficiency Classification & Regression

This project analyzes sleep health data to **predict and classify sleep efficiency** using modular Python code and machine learning.  
It is designed to provide a prediction tool usable even by individuals with limited data access, through a dual-model strategy.

---

## 📊 Project Objective

Sleep plays a critical role in brain detoxification and physical recovery. However, measuring sleep efficiency usually requires wearable devices or sleep labs.  
The goal of this project is to:

- Analyze factors that influence sleep efficiency
- Build classification and regression models using machine learning
- Offer a usable prediction tool for both data-rich and data-limited users

---

## 📂 Dataset Overview

- 📌 Source: Kaggle – Sleep Efficiency Dataset  
- 📌 Size: 452 samples → 438 after cleaning  
- 📌 Features: 15 original columns (sleep habits, lifestyle, time, demographics)  
- 🧪 Target: `Sleep efficiency` (0 to 1 float)

### Preprocessing Summary

- Removed multicollinear features:
  - `Sleep duration`: exactly = `Wakeup - Bedtime`
  - `Deep sleep %`: highly correlated with `Light sleep %`
- Categorical encoding:
  - `Gender`, `Smoking status` → binary (0/1)
- Outlier removal:
  - `Caffeine consumption`: used IQR method
- Missing value handling:
  - Median (for skewed vars), Mean (low-variance), stratified by gender
- Normalization:
  - Compared StandardScaler, MinMaxScaler, RobustScaler → chose StandardScaler

---

## ⚙️ Model Overview

### 🔠 Classification

- **Binary target**: `Good` if sleep efficiency ≥ 0.85, else `Poor`
- **Models**: Logistic Regression, Decision Tree, Random Forest
- **Validation**: Holdout, Stratified K-Fold (k=5)
- **Tuning**: GridSearchCV & RandomizedSearchCV
- **Final model**: Random Forest (RandomSearchCV, 5-Fold)

### 📈 Regression

- **Target**: sleep efficiency (continuous)
- **Models**: Linear Regression, Random Forest Regressor, Gradient Boosting
- **Final model**: Random Forest Regressor (RandomSearchCV, 5-Fold)

---

## 🧠 Real-World Strategy: Dual-Model Use

Because features like REM % and Light sleep % are difficult to obtain without wearables:

| Scenario                      | Features used             | Model type         |
|------------------------------|---------------------------|--------------------|
| User has full feature set    | All (incl. REM, Light)    | Advanced (accurate)|
| User lacks wearable device   | Partial (no REM/Light)    | Basic (practical)  |

This structure ensures flexibility and usability across different user contexts.

---

## 🧪 Scaler Experiment

Used both regression and classification tasks to compare:

- `StandardScaler` ✅
- `MinMaxScaler`
- `RobustScaler`

→ Minimal performance difference → chose **StandardScaler** for final use.

---

## 📁 Project Structure

```
datascience-termproject/
├── main.py                         # Runs all logic
├── src/
│   ├── preprocess.py               # Cleans and transforms data
│   ├── scaler_experiment.py        # Scaler comparison
│   ├── model_classification.py     # Classification workflow
│   └── model_regression.py         # Regression workflow
├── requirements.txt                # Dependencies
├── README.md
└── .gitignore
```

---

## ▶️ How to Run

1. Clone this repository:

```bash
git clone https://github.com/YeBeenChoi/DataScience_Termproject.git
cd datascience-termproject
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place the dataset file in the root folder:

```
ModelingSet.csv
```

4. Run the project:

```bash
python main.py
```

---

## 📦 Requirements

Install all dependencies with:

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

## 🔍 Learning Reflections

This project showed us that:

- There’s no single correct way to do data analysis — how we define the problem changes everything.
- Even with the same dataset, preprocessing choices, model selection, and evaluation strategies varied widely.
- Collaboration revealed blind spots and unlocked creative solutions.
- Practical constraints like user input limitations influenced final model design.

---

## 📜 License

Licensed under the **Apache License 2.0**.  
See the [LICENSE](./LICENSE) file for details.

