import os
import torch
import joblib
import pandas as pd
import config
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from dataset import CarsDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import CarPricePredictor
from utils import get_current_time, evaluate_model, make_directory, prepare_dataset, set_seed
from sklearn.model_selection import train_test_split


def train(model, opt, train_loader, test_loader, num_epochs, current_epoch=0, writer=None, criterion=None):

    if writer is None:
        writer = SummaryWriter(f"./tb/train/{get_current_time()}")

    if criterion is None:
        criterion = nn.MSELoss()

    best_score = evaluate_model(model, test_loader)

    # Цикл обучения:
    for epoch in range(current_epoch + 1, num_epochs + 1):

        model.train()  # Переключение модели в режим обучения

        epoch_loss = 0.0
        for idx, (x, y) in enumerate(tqdm(train_loader)):
            x = x.to(config.DEVICE)
            y = y.unsqueeze(1).to(config.DEVICE)

            # Получаем предсказания модели для текущего батча:
            y_pred = model(x).to(config.DEVICE)

            # Вычисляем loss:
            loss = criterion(y_pred, y)
            epoch_loss += loss

            # Обновляем веса модели:
            opt.zero_grad()
            loss.backward()
            opt.step()

        # После каждой эпохи тестируем модель:
        model.eval()  # Переключение модели в режим тестирования
        current_score = evaluate_model(model, test_loader)  # Тестирование

        # Обновляем tensorboard:
        writer.add_scalar("Score", current_score, global_step=epoch)  # Текущая точность модели
        writer.add_scalar("Loss", epoch_loss, global_step=epoch)  # Суммарный loss за текущую эпоху

        # Обновляем лучшую точность:
        if current_score < best_score:
            best_score = current_score

            # Сохраняем чекпоинт с лучшей точностью, если необходимо:
            if config.SAVE_BEST_MODEL:
                print("\033[32m=> Сохранение чекпоинта\033[0m")

                # Создаем директорию для сохранения
                dir_name = get_current_time()
                dir_path = os.path.join(config.TRAIN_DIR, dir_name)
                make_directory(dir_path)

                # Сохраняем
                model_path = os.path.join(dir_path, config.CHECKPOINT_NAME)
                # save_checkpoint(model, opt, model_path, epoch)

    writer.close()

    return best_score


def main():
    set_seed(seed=config.RANDOM_STATE)

    # Создаем директорию текущего обучения
    train_dir_name = f"train_{get_current_time()}"
    train_dir_path = os.path.join(config.TRAIN_DIR, train_dir_name)
    make_directory(train_dir_path)

    # Инициализируем модель:
    model = CarPricePredictor()
    model = model.to(config.DEVICE)

    # Определяем параметры, которые будем обновлять при обучении:
    params_to_update = [param for param in model.parameters() if param.requires_grad]

    # Инициализируем оптимизатор:
    opt = optim.Adam(params=params_to_update, lr=config.LEARNING_RATE)

    # Загружаем датасет:
    dataset = pd.read_csv("./data/cars_dataset.csv")
    dataset, feature_preprocessor, target_scaler = prepare_dataset(dataset)

    joblib.dump(feature_preprocessor, f"{train_dir_path}/feature_preprocessor.joblib")
    joblib.dump(target_scaler, f"{train_dir_path}/target_scaler.joblib")

    train_data, test_data = train_test_split(dataset,
                                             test_size=0.25,
                                             shuffle=True,
                                             random_state=config.RANDOM_STATE)

    train_dataset = CarsDataset(train_data)
    test_dataset = CarsDataset(test_data)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    current_epoch = 0  # Текущая эпоха обучения

    # Загружаем последний чекпоинт модели:
    if config.LOAD_MODEL:
        print("\033[32m=> Загрузка последнего чекпоинта\033[0m")

        # checkpoint_path = get_last_checkpoint()
        # model, opt, current_epoch = load_checkpoint(model, opt, checkpoint_path)

    num_epochs = config.NUM_EPOCHS  # Количество эпох обучения

    # Обучаем модель:
    train(model, opt, train_loader, test_loader, num_epochs, current_epoch=current_epoch)


if __name__ == "__main__":
    main()
