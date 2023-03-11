import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader
import config
import optuna
from dataset import CarsDataset
from train import train
from model import CarPricePredictor
from utils import prepare_dataset


def objective(trial):
    # Гиперпараметры, которые будем оптимизировать:
    batch_size = trial.suggest_int("batch_size", 1, 128)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

    # Инициализируем модель:
    model = CarPricePredictor(input_size=config.INPUT_FEATURES, output_size=config.OUT_FEATURES)

    model = model.to(config.DEVICE)

    # Определяем параметры, которые будем обновлять при обучении:
    params_to_update = [param for param in model.parameters() if param.requires_grad]

    # Инициализируем оптимизатор:
    opt = optim.Adam(params=params_to_update, lr=learning_rate)

    # Загружаем датасет:
    dataset = pd.read_csv("./data/cars_dataset.csv")
    dataset, feature_preprocessor, target_scaler = prepare_dataset(dataset)

    train_data, test_data = train_test_split(dataset,
                                             test_size=0.25,
                                             shuffle=True,
                                             random_state=config.RANDOM_STATE)

    train_dataset = CarsDataset(train_data)
    test_dataset = CarsDataset(test_data)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    current_epoch = 0  # Текущая эпоха обучения

    num_epochs = 30  # Количество эпох обучения

    # Обучаем модель:
    score = train(model, opt, train_loader, test_loader, num_epochs, current_epoch=current_epoch)

    return score


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    print("Best score: ", study.best_value)
    print("Best params: ", study.best_params)
