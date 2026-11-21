# Kaggle Titanic — ML Pipeline

Учебный модульный ML-пайплайн для задачи бинарной классификации [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic). Проект объединяет разведочный анализ данных, предобработку признаков, валидацию нескольких семейств моделей (классический ML, градиентный бустинг, ансамбль, нейросеть на PyTorch) и генерацию файла `submission.csv` через YAML-конфигурацию.

## Структура проекта

```text
titanic_project/
├── data/
│   └── raw/                  # Исходные файлы Kaggle (train.csv, test.csv — в .gitignore)
├── notebooks/
│   └── 01_eda.ipynb          # Разведочный анализ данных
├── src/
│   ├── __init__.py
│   ├── data.py               # Загрузка CSV-файлов
│   ├── preprocess.py         # Очистка, импутация пропусков, feature engineering и кодирование
│   ├── dl.py                 # PyTorch Dataset (TitanicDataset) и архитектура MLP (TitanicNN)
│   ├── models.py             # Фабрика моделей (LogReg, RandomForest, CatBoost, VotingClassifier)
│   └── train.py              # Stratified K-Fold CV, GridSearchCV, обучение MLP и инференс
├── config.yaml               # Пути к данным, параметры CV, активная модель и сетки параметров
├── main.py                   # Точка входа в пайплайн
├── requirements.txt          # Зависимости проекта
└── .gitignore
```

## Основные этапы пайплайна

1. **Предобработка и Feature Engineering (`src/preprocess.py`):**
   - Удаление технических и сильно разреженных столбцов (`PassengerId`, `Ticket`, `Cabin`).
   - Заполнение пропусков `Age` и `Fare` медианой внутри классов каюты (`Pclass`), а `Embarked` — модой.
   - Генерация новых признаков: `FamilySize`, `IsAlone` и `Title` (извлечение титула из `Name` с группировкой редких категорий в `Rare`).
   - Кодирование категорий (`Sex`, `Embarked`, `Title`): нативный тип `category` для CatBoost и `pd.get_dummies` для остальных алгоритмов.
2. **Модели (`src/models.py`, `src/dl.py`):**
   - `logistic_regression` — логистическая регрессия (`scikit-learn`).
   - `random_forest` — случайный лес (`scikit-learn`).
   - `catboost` — градиентный бустинг с нативной поддержкой категорий (`CatBoostClassifier`).
   - `voting_ensemble` — мягкое голосование (`VotingClassifier`) поверх `LogisticRegression`, `RandomForestClassifier` и `CatBoostClassifier` со `StandardScaler`.
   - `neural_network` — двухслойный MLP на PyTorch (`Linear -> ReLU -> Linear`, `BCEWithLogitsLoss`, `Adam`, масштабирование `Age`, `Fare`, `FamilySize`).
3. **Валидация и инференс (`src/train.py`, `main.py`):**
   - Оценка качества на 5-Fold `StratifiedKFold` (метрика `Accuracy`), подбор гиперпараметров через `GridSearchCV` для классических моделей.
   - Обучение финальной модели на полной выборке и сохранение предсказаний в `submission.csv`.

## Установка и запуск

1. Создайте виртуальное окружение и установите зависимости:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Поместите `train.csv` и `test.csv` со страницы соревнования Kaggle в папку `data/raw/`.
3. Укажите нужную модель в поле `active_model` файла `config.yaml` (`logistic_regression`, `random_forest`, `catboost`, `neural_network` или `voting_ensemble`).
4. Запустите пайплайн:
   ```bash
   python main.py --config config.yaml
   ```

## Результаты экспериментов (5-Fold Stratified CV, Accuracy)

| Модель | CV Accuracy | Public LB (Kaggle) |
| :--- | :---: | :---: |
| Logistic Regression | `0.XXXX` | — |
| Random Forest | `0.XXXX` | — |
| CatBoost | `0.XXXX` | — |
| PyTorch MLP | `0.XXXX` | — |
| Voting Ensemble | `0.XXXX` | `0.XXXX` |
