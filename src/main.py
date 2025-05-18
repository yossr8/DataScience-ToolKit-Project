from src.loading_data.load_data import load, feature_engineering, split_data, build_preprocessing_pipeline
from src.train import train_models

def main():
    df = load("G:/ITI DataScience 9Months/mlops/mlopsproject/MlopsProject/data/raw/train.csv")
    df_fe = feature_engineering(df)
    x_train, x_test, y_train, y_test = split_data(df_fe, target_column="Survived")
    preprocessor = build_preprocessing_pipeline(x_train)

    results = train_models(x_train, x_test, y_train, y_test, preprocessor)
    print("\nTraining complete.")
    print(results)

if __name__ == "__main__":
    main()