import yaml
import argparse
from src.data import load_data
from src.preprocess import preprocess_pipeline
from src.train import run_validation, get_final_prediction
from src.models import get_model
import pandas as pd


def load_config(config_path: str) -> dict:
    """Читает YAML-файл с параметрами эксперимента и возвращает словарь настроек."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    """Точка входа: загружает данные, запускает препроцессинг, валидацию и формирует submission.csv."""
    parser = argparse.ArgumentParser(description="Titanic ML Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Pipeline is running. Model: {config['active_model']}")

    # 1. Загрузка обучающей выборки
    df_train = load_data(config['data']['raw_train_path'])
    print("Step 1: Data loading complete")

    # 2. Предобработка и генерация признаков
    df_train_processed = preprocess_pipeline(
        df_train, model_name=config['active_model'], is_train=True
    )
    print("Step 2: Preprocessing complete")

    # 3. Кросс-валидация и подбор гиперпараметров
    run_validation(df_train_processed, config)
    print("Step 3: Validation and training complete")

    # 4. Подготовка тестовых данных, финальное обучение и сохранение сабмита
    print("Step 4: Creating submission")
    df_test = load_data(config['data']['raw_test_path'])
    test_ids = df_test['PassengerId'].copy()

    df_test_processed = preprocess_pipeline(
        df_test, model_name=config['active_model'], is_train=True
    )
    X_test = df_test_processed
    X_train_full = df_train_processed.drop(columns=['Survived'])
    y_train_full = df_train_processed['Survived']

    preds = get_final_prediction(X_train_full, y_train_full, X_test, config)

    submission = pd.DataFrame({
        'PassengerId': test_ids,
        'Survived': preds
    })
    submission.to_csv("submission.csv", index=False)
    print("Step 4: Submission saved to submission.csv")


if __name__ == "__main__":
    main()
