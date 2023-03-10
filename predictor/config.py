import torch

# Предустановки
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_FEATURES = 1575
OUT_FEATURES = 1
NUM_WORKERS = 2
RANDOM_STATE = 42

# Обучение
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 3e-4

LOAD_MODEL = False
SAVE_BEST_MODEL = False

# Датасет
DATASET_FILE = "./data/cars_dataset.csv"

# Другое
TRAIN_DIR = "trains"
CHECKPOINT_NAME = "model.pth.tar"
