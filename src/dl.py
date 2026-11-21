import torch
import torch.nn as nn
from torch.utils.data import Dataset


class TitanicDataset(Dataset):
    """PyTorch Dataset для табличных данных Titanic.

    Поддерживает режим обучения и режим инференса
    """

    def __init__(self, X_values, y_values=None):
        self.X_values = torch.tensor(X_values.to_numpy(), dtype=torch.float32)
        if y_values is not None:
            self.y_values = torch.tensor(y_values.to_numpy(), dtype=torch.float32).unsqueeze(1)
        else:
            self.y_values = None

    def __len__(self):
        return self.X_values.size(dim=0)

    def __getitem__(self, idx):
        if self.y_values is None:
            return self.X_values[idx]
        else:
            return self.X_values[idx], self.y_values[idx]


class TitanicNN(nn.Module):
    """Двухслойный полносвязный перцептрон (MLP) для бинарной классификации.

    Возвращает сырые логиты для использования с BCEWithLogitsLoss.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.pipeline = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.pipeline(x)
