import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from src.models import get_model
from torch.utils.data import DataLoader
from src.dl import TitanicDataset, TitanicNN
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
import torch.nn as nn
import torch


def train_nn(X_train, y_train, X_val, y_val, config):
    """Обучает и валидирует PyTorch MLP на одном фолде.

    Масштабирует непрерывные признаки (Age, Fare, FamilySize) по train-фолду,
    сохраняет веса с наилучшей точностью в best_model.pth и возвращает лучшую accuracy.
    """
    batch_size = config['models']['neural_network']['batch_size']
    epochs = config['models']['neural_network']['epochs']
    lr = config['models']['neural_network']['lr']
    hidden_dim = config['models']['neural_network']['hidden_dim']

    # Масштабирование непрерывных признаков внутри фолда
    scaler = StandardScaler()
    columns_to_scale = ['Age', 'Fare', 'FamilySize']

    X_train = X_train.copy()
    X_val = X_val.copy()

    X_train[columns_to_scale] = X_train[columns_to_scale].astype(float)
    X_val[columns_to_scale] = X_val[columns_to_scale].astype(float)

    X_train.loc[:, columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    X_val.loc[:, columns_to_scale] = scaler.transform(X_val[columns_to_scale])

    train_data = TitanicDataset(X_train, y_train)
    val_data = TitanicDataset(X_val, y_val)

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = TitanicNN(input_dim=X_train.shape[1], hidden_dim=hidden_dim).to(device)

    loss_f = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_accuracy = 0.0

    for epoch in range(epochs):
        # Этап обучения
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_f(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Этап валидации
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_f(outputs, labels)
                val_loss += loss.item()

                # Логит > 0 эквивалентен вероятности > 0.5 после Sigmoid
                preds = (outputs > 0.0).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        current_accuracy = correct / total
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            torch.save(model.state_dict(), 'best_model.pth')

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(
                f"Epoch: {epoch+1}/{epochs}, "
                f"Train Loss: {train_loss/len(train_loader):.4f} | "
                f"Val Loss: {val_loss/len(val_loader):.4f} | "
                f"Accuracy: {current_accuracy:.4f}"
            )

    print(f"Best Accuracy: {best_accuracy:.4f}")
    return best_accuracy


def run_validation(df: pd.DataFrame, config: dict) -> float:
    """Запускает Stratified K-Fold кросс-валидацию для активной модели.

    Для neural_network выполняет цикл обучения по фолдам через train_nn,
    для классических моделей и ансамбля запускает GridSearchCV по сетке из config.yaml.
    """
    X = df.drop(columns=['Survived'])
    y = df['Survived']

    n_splits = config['validation']['n_splits']
    random_state = config['validation']['random_state']

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model_name = config['active_model']

    if model_name == 'neural_network':
        print("Lauch Neural Network Cross-Validation...")
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n--- Fold {fold + 1} ---")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            score = train_nn(X_train, y_train, X_val, y_val, config)
            fold_scores.append(score)

        print(f"\nFolds accuracy: {fold_scores}")
        mean_acc = np.mean(fold_scores)
        print(f"Mean Accuracy: {mean_acc:.4f}")
        return mean_acc

    model = get_model(config)
    param_grid = config['models'][model_name]

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1
    )

    print("Lauch GridSearchCV")
    if model_name == 'catboost':
        grid.fit(X, y, cat_features=['Sex', 'Embarked', 'Title'])
    else:
        grid.fit(X, y)

    print(f"Best Accuracy: {grid.best_score_:.4f}")
    print(f"Best Params: {grid.best_params_}")

    return grid.best_score_


def get_final_prediction(X_train, y_train, X_test, config) -> list[int]:
    """Обучает финальную модель на всей обучающей выборке и возвращает предсказания для теста."""
    model_name = config['active_model']
    if model_name == 'neural_network':
        scaler = StandardScaler()
        columns_to_scale = ['Age', 'Fare', 'FamilySize']
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        X_train_scaled[columns_to_scale] = X_train_scaled[columns_to_scale].astype(float)
        X_test_scaled[columns_to_scale] = X_test_scaled[columns_to_scale].astype(float)

        X_train_scaled.loc[:, columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
        X_test_scaled.loc[:, columns_to_scale] = scaler.transform(X_test[columns_to_scale])

        batch_size = config['models']['neural_network']['batch_size']
        epochs = config['models']['neural_network']['epochs']
        lr = config['models']['neural_network']['lr']
        hidden_dim = config['models']['neural_network']['hidden_dim']

        train_data = TitanicDataset(X_train_scaled, y_train)
        test_data = TitanicDataset(X_test_scaled)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = TitanicNN(input_dim=X_train.shape[1], hidden_dim=hidden_dim).to(device)
        loss_f = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        print(f"Обучение финальной нейросети на всех данных ({epochs} эпох)...")
        for epoch in range(epochs):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_f(outputs, labels)
                loss.backward()
                optimizer.step()

        model.eval()
        preds_list = []
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                batch_preds = (outputs > 0.0).float().cpu().numpy().flatten()
                preds_list.extend(batch_preds)

        return [int(p) for p in preds_list]

    else:
        final_model = get_model(config)
        best_params = {k: v[0] for k, v in config['models'][model_name].items()}
        final_model.set_params(**best_params)

        if model_name == 'catboost':
            final_model.fit(X_train, y_train, cat_features=['Sex', 'Embarked', 'Title'])
        else:
            final_model.fit(X_train, y_train)

        raw_preds = final_model.predict(X_test)
        return [int(p) for p in raw_preds]
