from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

print("train.py loaded")
def train_models(x_train, x_test, y_train, y_test, preprocessor, save_path="models/"):
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining: {name}")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])
        cv_scores = cross_val_score(pipeline, x_train, y_train, cv=5, scoring="accuracy")
        print(f"CV Accuracy (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        pipeline.fit(x_train, y_train)

        y_pred = pipeline.predict(x_test)
        test_acc = accuracy_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test MSE: {mse:.4f}")

        joblib.dump(pipeline, f"{save_path}{name}.pkl")

        results[name] = {
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "test_accuracy": test_acc,
            "test_mse": mse,
        }

    return results
