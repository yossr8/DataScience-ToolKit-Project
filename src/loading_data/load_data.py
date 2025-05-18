import pandas as pd
import joblib
import os 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load(path:str)-> pd.DataFrame:
    if os.path.exists(path):
        train = pd.read_csv(path)
    else:
        raise FileNotFoundError(f"file in {path} dos not exist")
    return train 

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()

def split_data(df: pd.DataFrame, target_column: str):
    x = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(x, y, test_size=0.2, random_state=42)

def build_preprocessing_pipeline(x: pd.DataFrame) -> ColumnTransformer:
    numeric_features = ["Age", "Fare", "SibSp", "Parch", "Pclass"]
    categorical_features = ["Sex", "Embarked"]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    return preprocessor





     