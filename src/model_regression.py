import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint

def run_regression(df):
    X = df.drop(columns=['Sleep efficiency'])
    y = df['Sleep efficiency']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    def evaluate_test(model, model_name, kfold_desc):
        model.fit(X_train_scaled, y_train)
        y_pred_test = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        results.append({
            'KFold': kfold_desc, 'Model': model_name, 'Stage': 'Test',
            'MSE': mse, 'R²': r2
        })

    def evaluate_cv(model, model_name, kf, kfold_desc):
        y_pred_cv = cross_val_predict(model, X_train_scaled, y_train, cv=kf)
        mse = mean_squared_error(y_train, y_pred_cv)
        r2 = r2_score(y_train, y_pred_cv)
        results.append({
            'KFold': kfold_desc, 'Model': model_name, 'Stage': 'CV',
            'MSE': mse, 'R²': r2
        })

    models = [
        ("Linear Regression", LinearRegression()),
        ("Random Forest Regressor", RandomForestRegressor(random_state=42)),
        ("Gradient Boosting Regressor", GradientBoostingRegressor(random_state=42))
    ]

    for name, model in models:
        evaluate_test(model, name, "Test")
        for k in [3, 5, 10]:
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
            evaluate_cv(model, name, kf, f"{k}-Fold")

    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }

    param_dist = {
        'n_estimators': randint(50, 200),
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10]
    }

    for k in [3, 5, 10]:
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=kf, n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        best_rf_grid = grid.best_estimator_
        evaluate_test(best_rf_grid, f"Random Forest (GridSearch, {k}-Fold)", "Test")
        evaluate_cv(best_rf_grid, f"Random Forest (GridSearch, {k}-Fold)", kf, f"{k}-Fold")

        random_search = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_distributions=param_dist, n_iter=20, cv=kf, random_state=42, n_jobs=-1)
        random_search.fit(X_train_scaled, y_train)
        best_rf_random = random_search.best_estimator_
        evaluate_test(best_rf_random, f"Random Forest (RandomSearch, {k}-Fold)", "Test")
        evaluate_cv(best_rf_random, f"Random Forest (RandomSearch, {k}-Fold)", kf, f"{k}-Fold")

    return pd.DataFrame(results)

