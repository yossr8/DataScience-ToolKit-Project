from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
from hydra.utils import instantiate
import os

def train_models(x_train, x_test, y_train, y_test, preprocessor, models_cfg, training_cfg):
    results = {}

    save_path = training_cfg.save_path
    os.makedirs(save_path, exist_ok=True)

    for name, model_cfg in models_cfg.items():
        print(f"\nTraining: {name}")

        # Instantiate model from Hydra config
        model = instantiate(model_cfg)

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        cv_scores = cross_val_score(pipeline, x_train, y_train, cv=training_cfg.cv_folds, scoring="accuracy")
        print(f"CV Accuracy (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)

        test_acc = accuracy_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test MSE: {mse:.4f}")

        joblib.dump(pipeline, f"{save_path}/{name}.pkl")

        results[name] = {
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "test_accuracy": test_acc,
            "test_mse": mse,
        }

    return results