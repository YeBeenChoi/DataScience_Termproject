import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from scipy.stats import randint

def run_classification(df):
    df['Sleep_Label'] = (df['Sleep efficiency'] >= 0.85).astype(int)
    X = df.drop(columns=['Sleep efficiency', 'Sleep_Label'])
    y = df['Sleep_Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    def evaluate_test(model, model_name, stage):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        results.append({
            'KFold': 'Test', 'Model': model_name, 'Stage': stage,
            'Accuracy': accuracy_score(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_proba),
            'R²': r2_score(y_test, y_proba)
        })

    def evaluate_cv(model, model_name, n_fold):
        skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
        y_proba = cross_val_predict(model, X_train_scaled, y_train, cv=skf, method="predict_proba")[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        results.append({
            'KFold': f"{n_fold}-Fold", 'Model': model_name, 'Stage': 'CV',
            'Accuracy': accuracy_score(y_train, y_pred),
            'MSE': mean_squared_error(y_train, y_proba),
            'R²': r2_score(y_train, y_proba)
        })

    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("Decision Tree", DecisionTreeClassifier(random_state=42)),
        ("Random Forest", RandomForestClassifier(random_state=42))
    ]

    for name, model in models:
        evaluate_test(model, name, "Test")
        for k in [3, 5, 10]:
            evaluate_cv(model, name, k)

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
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=skf, n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        best_grid = grid.best_estimator_
        evaluate_test(best_grid, f"Random Forest (GridSearch, {k}-Fold)", "Test")
        evaluate_cv(best_grid, f"Random Forest (GridSearch, {k}-Fold)", k)

        random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=param_dist, n_iter=20, cv=skf, random_state=42, n_jobs=-1)
        random_search.fit(X_train_scaled, y_train)
        best_random = random_search.best_estimator_
        evaluate_test(best_random, f"Random Forest (RandomSearch, {k}-Fold)", "Test")
        evaluate_cv(best_random, f"Random Forest (RandomSearch, {k}-Fold)", k)

    return pd.DataFrame(results)

