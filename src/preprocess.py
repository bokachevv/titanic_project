import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет технические и малоинформативные столбцы с большим числом пропусков."""
    df = df.copy()
    df = df.drop(columns=['PassengerId', 'Ticket', 'Cabin'])
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Заполняет пропуски в числовых и категориальных признаках.

    Для Age и Fare используется медиана внутри каждого класса каюты (Pclass),
    для Embarked - мода
    """
    df = df.copy()
    df['Age'] = df['Age'].fillna(df.groupby('Pclass')['Age'].transform('median'))
    df['Fare'] = df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('median'))
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    return df


def encode_categorical(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Кодирует категориальные признаки в зависимости от выбранной модели.

    Для CatBoost столбцы приводятся к типу category для нативной обработки,
    для остальных моделей применяется One-Hot Encoding (get_dummies).
    """
    df = df.copy()
    if model_name != 'catboost':
        df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title'], dtype=int)
    else:
        df['Sex'] = df['Sex'].astype('category')
        df['Embarked'] = df['Embarked'].astype('category')
        df['Title'] = df['Title'].astype('category')

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Генерирует новые признаки семьи и социального статуса (титула) из Name."""
    df = df.copy()

    # Размер семьи и бинарный флаг одиночного пассажира
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Извлечение титула из строки имени и объединение редких/синонимичных категорий
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
        'Rare'
    )
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace(['Mme'], 'Mrs')
    df['Title'] = df['Title'].fillna('Rare')
    return df


def preprocess_pipeline(df: pd.DataFrame, model_name: str, is_train: bool = True) -> pd.DataFrame:
    """Запускает полный конвейер предобработки датафрейма для указанной модели.

    Args:
        df: Исходный DataFrame.
        model_name: Имя активной модели из config.yaml.
        is_train: Флаг режима обработки.

    Returns:
        Подготовленный DataFrame с закодированными признаками.
    """
    df = clean_data(df)
    df = impute_missing(df)
    df = create_features(df)

    # После извлечения Title исходный строковый столбец Name больше не нужен
    if 'Name' in df.columns:
        df = df.drop(columns=['Name'])

    df = encode_categorical(df, model_name)

    return df
