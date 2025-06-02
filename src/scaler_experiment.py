import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

def run_scaler_experiments(df):
    X = df.drop('Sleep efficiency', axis=1)
    y_reg = df['Sleep efficiency']
    y_cls = (y_reg >= 0.85).astype(int)

    scalers = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler()
    }

    print('=== Regression (Linear Regression) ===')
    for name, scaler in scalers.items():
        pipe = Pipeline([
            ('scaler', scaler),
            ('model', LinearRegression())
        ])
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rmse = -cross_val_score(pipe, X, y_reg, cv=kf, scoring='neg_root_mean_squared_error')
        r2  =  cross_val_score(pipe, X, y_reg, cv=kf, scoring='r2')
        print(f'\n{name}:')
        print(f'  Mean RMSE: {rmse.mean():.4f} ± {rmse.std():.4f}')
        print(f'  Mean R²:   {r2.mean():.4f} ± {r2.std():.4f}')

    print('\n=== Classification (Random Forest) ===')
    for name, scaler in scalers.items():
        pipe = Pipeline([
            ('scaler', scaler),
            ('model', RandomForestClassifier(random_state=42))
        ])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        acc = cross_val_score(pipe, X, y_cls, cv=skf, scoring='accuracy')
        print(f'\n{name}:')
        print(f'  Mean Accuracy: {acc.mean():.4f} ± {acc.std():.4f}')

