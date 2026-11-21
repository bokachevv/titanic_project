from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_model(config: dict):
    """Возвращает экземпляр модели по ключу active_model из конфигурации.

    Поддерживает logistic_regression, random_forest, catboost и voting_ensemble
    (мягкое голосование поверх LogisticRegression, RandomForest и CatBoost с масштабированием).
    """
    model_name = config['active_model']
    model_params = config['models'][model_name]

    if model_name == 'logistic_regression':
        return LogisticRegression()
    elif model_name == 'random_forest':
        return RandomForestClassifier()
    elif model_name == 'catboost':
        return CatBoostClassifier()
    elif model_name == 'voting_ensemble':
        lr = LogisticRegression(C=1.0, max_iter=1000)
        rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
        cb = CatBoostClassifier(
            iterations=300, depth=6, l2_leaf_reg=10, learning_rate=0.1, verbose=0, random_state=42
        )

        ensemble = VotingClassifier(
            estimators=[('lr', lr), ('rf', rf), ('cb', cb)],
            voting='soft'
        )
        return Pipeline([('scaler', StandardScaler()), ('model', ensemble)])
    else:
        raise ValueError(f"Unknown model: {model_name}")
