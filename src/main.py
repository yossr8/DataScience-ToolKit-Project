from src.loading_data.load_data import load, feature_engineering, split_data, build_preprocessing_pipeline
from src.train import train_models
import hydra
import pandas as pd
from typing import cast
from omegaconf import DictConfig

@hydra.main(config_path="../configs", config_name="config", version_base=None)

def main(cfg: DictConfig):
    print("Config loaded:\n", cfg)
    df = load(cfg.data.path)
    df_fe = feature_engineering(df)
    x_train, x_test, y_train, y_test = split_data(df_fe, target_column=cfg.data.target_column)
    x_train = cast(pd.DataFrame, x_train)
    preprocessor = build_preprocessing_pipeline(x_train)
    results = train_models(
        x_train, x_test, y_train, y_test,
        preprocessor,
        models_cfg=cfg.models,
        training_cfg=cfg.training
    )
    print("Training Results:\n", results)

if __name__ == "__main__":
    print("Starting training...")
    main()