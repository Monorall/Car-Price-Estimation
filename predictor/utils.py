import os
import random
import sys

import torch
import shutil
import config
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from joblib import dump


def make_directory(dir_path: str) -> None:
    """Создаёт директорию. Если директория существует - перезаписывает."""

    try:
        os.makedirs(dir_path)
    except FileExistsError:
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)


def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def evaluate_model(model, data_loader):
    # Set the model to evaluation mode
    model.eval()

    # Move the model to the specified device
    model.to(config.DEVICE)

    # Track the total MSE of the model on the test dataset
    total_mse = 0

    # Iterate over the test data
    with torch.no_grad():
        for inputs, targets in data_loader:
            # Move the inputs and targets to the specified device
            inputs, targets = inputs.to(config.DEVICE), targets.unsqueeze(1).to(config.DEVICE)

            # Forward pass through the model
            outputs = model(inputs)

            # Compute the MSE of the model on this batch of test data
            mse = torch.nn.functional.mse_loss(outputs, targets)

            # Add the MSE of this batch to the total MSE
            total_mse += mse.item()

    # Compute the mean squared error of the model on the test dataset
    mean_mse = total_mse / len(data_loader)

    return mean_mse


def prepare_dataset(df):
    cat_vars = ["cleared_customs", "brand", "model", "car_body",
                "color", "transmission_type", "drive_type", "fuel_type",
                "exterior_condition", "after_an_accident", "fine_condition",
                "first_owner", "garage_storage", "needs_body_repair", "needs_engine_repair",
                "needs_undercarriage_repair", "not_bit", "not_colored", "not_on_the_move"]

    num_vars = ["motor_mileage_thou", "motor_engine_size_litre", "age"]

    target_var = ["price"]

    feature_encoder = OneHotEncoder(sparse_output=False)
    feature_scaler = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", feature_encoder, cat_vars),
            ("num", feature_scaler, num_vars)
        ]
    )

    target_scaler = StandardScaler()
    scaled_target = target_scaler.fit_transform(df[target_var])
    target_df = pd.DataFrame(scaled_target, columns=target_var)

    features = preprocessor.fit_transform(df)
    feature_names = [f"{var}_{cat}" for var, cats in zip(cat_vars, preprocessor.named_transformers_['cat'].categories_) for cat in cats]
    feature_df = pd.DataFrame(features, columns=feature_names + num_vars)

    prepared_df = pd.concat([feature_df, target_df], axis=1)

    return prepared_df, preprocessor, target_scaler


def save_checkpoint(model, optimizer, model_path, epoch=0) -> None:
    """Сохраняет чекпоинт модели в процессе обучения (модель, оптимизатор, номер эпохи)."""

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, model_path)


def load_checkpoint(model, optimizer, checkpoint_file):
    """Загружает чекпоинт модели. Возвращает модель, оптимизатор, номер эпохи"""

    if not os.path.isfile(checkpoint_file):
        raise FileNotFoundError(f"Ошибка: не удалось найти {checkpoint_file}")

    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]

    return model, optimizer, epoch


def get_last_checkpoint() -> str:
    """Возвращает путь к последнему по времени сохранённому чекпоинту."""

    try:
        checkpoints = [d for d in os.listdir(config.TRAIN_DIR) if os.path.isdir(os.path.join(config.TRAIN_DIR, d))]
        if not checkpoints:
            raise IndexError
        checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(config.TRAIN_DIR, x)))  # Сортировка по времени
        path_to_model = os.path.join(config.TRAIN_DIR, checkpoints[-2], config.CHECKPOINT_NAME)
        return path_to_model
    except IndexError:
        print(f"Ошибка: в директории {config.TRAIN_DIR} нет сохраненных чекпоинтов")
        sys.exit(1)
    except FileNotFoundError:
        print(f'Ошибка: не удалось загрузить {config.CHECKPOINT_NAME}')
        sys.exit(1)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Random seed set as {seed}")
