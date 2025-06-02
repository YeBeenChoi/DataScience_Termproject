import pandas as pd
from src.preprocess import preprocess
from src.scaler_experiment import run_scaler_experiments
from src.model_classification import run_classification
from src.model_regression import run_regression

if __name__ == "__main__":
    df = pd.read_csv("ModelingSet.csv")
    df = preprocess(df)

    run_scaler_experiments(df)

    print("\n=== Classification Results ===")
    classification_results = run_classification(df)
    print(classification_results)

    print("\n=== Regression Results ===")
    regression_results = run_regression(df)
    print(regression_results)

