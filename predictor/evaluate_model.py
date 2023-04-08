import json
import torch
import config
import joblib
import numpy as np
import pandas as pd
from utils import *
from joblib import load
from torch import optim
from model import CarPricePredictor


def prepare_input(data_json, preprocessor_path):
    # Load preprocessor from saved file
    preprocessor = joblib.load(preprocessor_path)

    # Extract columns from dataframe
    cat_vars = ["cleared_customs", "brand", "model", "car_body",
                "color", "transmission_type", "drive_type", "fuel_type",
                "exterior_condition", "after_an_accident", "fine_condition",
                "first_owner", "garage_storage", "needs_body_repair", "needs_engine_repair",
                "needs_undercarriage_repair", "not_bit", "not_colored", "not_on_the_move"]
    num_vars = ["motor_mileage_thou", "motor_engine_size_litre", "age"]

    # Создание DataFrame с сохранением порядка столбцов
    columns = cat_vars + num_vars
    df = pd.DataFrame(data_json, columns=columns)

    # Apply preprocessor to features
    features = preprocessor.transform(df)

    return torch.tensor(features[0], dtype=torch.float32)


def process_output(price, target_scaler_path):
    # Load target_scaler from saved file
    target_scaler = joblib.load(target_scaler_path)

    # Scale target variable
    scaled_price = target_scaler.inverse_transform(price.reshape(-1, 1))

    return scaled_price[0][0]


def evaluate(features_json: str) -> float:
    features = json.loads(features_json)

    model = CarPricePredictor(input_size=config.INPUT_FEATURES, output_size=config.OUT_FEATURES)
    opt = optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)

    model, _, _ = load_checkpoint(model, opt, f"./checkpoint/{config.CHECKPOINT_NAME}")

    model.eval()

    input = prepare_input(features, "./checkpoint/feature_preprocessor.joblib")
    output = model(input)

    processed_output = process_output(output.detach().numpy(), "./checkpoint/target_scaler.joblib")

    return processed_output


# if __name__ == "__main__":
#     set_seed()
#
#     #  "price": 100.0
#     json_data = '[{"cleared_customs":true,"brand":"Opel","model":"corsa","motor_mileage_thou":170.0,"car_body":"hatchback","color":"red","transmission_type":"manual","drive_type":"front","fuel_type":"diesel","motor_engine_size_litre":1.3,"exterior_condition":"major_fixes_needed","lat":49.90548,"lon":24.09005,"after_an_accident":true,"fine_condition":false,"first_owner":false,"garage_storage":false,"needs_body_repair":false,"needs_engine_repair":false,"needs_undercarriage_repair":false,"not_bit":false,"not_colored":false,"not_on_the_move":false,"age":12.0}]'
#
#     json_data = json.loads(json_data)
#     # Преобразование строки JSON в объект Python
#     data = json.loads(json_data)
#
#     output = evaluate(features=data)
#
#     print(output)

