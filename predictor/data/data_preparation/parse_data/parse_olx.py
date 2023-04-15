import requests
import pandas as pd

# Список параметров для одного объявления. Также список названий столбцов в конечном датафрейме
fields = ['id', 'created_time', 'car_state_type',
          'cleared_customs', 'brand', 'model', 'motor_year', 'motor_mileage_thou',
          'car_body', 'color', 'transmission_type', 'drive_type', 'fuel_type',
          'motor_engine_size_litre', 'exterior_condition', 'condition', 'region', 'lat', 'lon', 'price']


def get_json_data(request: str):
    """ Принимает на вход запрос, при удачном выполнении возвращает полученный json, иначе None. """

    response = requests.get(request)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error:", response)
        return None


def prepare_json(data_json: dict):
    """ Принимает на вход json, возвращает dataframe"""

    #  Находим номера органических (не продвигаемых) объявлений в json.
    #  Если json не содержит объявлений, возвращаем пустой dataframe:
    try:
        organic_car_numbers = data_json["metadata"]["source"]["organic"]
    except KeyError:
        return pd.DataFrame()

    brands = pd.read_csv("brands.csv").set_index('id').to_dict()['brand']  # Бренды автомобилей

    cars = []  # Пустой список для сохранения результатов в виде словарей (dict)

    # Достаем данные об автомобиле и объявлении из json'а. Отсутствующие заполняем как None:
    for num in organic_car_numbers:
        current_car = {k: None for k in fields}

        current_car["id"] = data_json["data"][num].get("id")  # Id объявления
        current_car["created_time"] = data_json["data"][num].get("created_time")  # Время создания

        # Бренд авто
        brand_id = data_json["data"][num]["category"].get("id")
        current_car["brand"] = brands.get(brand_id)

        # Регион
        current_car["region"] = data_json["data"][num]["location"]["region"].get("normalized_name")

        # Широта и долгота
        loc = data_json["data"][num]["map"]
        current_car["lat"] = loc.get("lat")
        current_car["lon"] = loc.get("lon")

        car_params = data_json["data"][num]["params"]  # Неупорядоченный список всех параметров автомобиля

        for param in car_params:
            if param["key"] in fields:

                # Пробуем получить значения для текущего параметра:
                key_value = param["value"].get("key")  # Все параметры кроме типа трансмиссии, типа топлива и цены
                label_value = param["value"].get("label")  # Трансмиссия и тип топлива
                value = param["value"].get("value")  # Цена

                # Ассоциируем значение с соответствующим ключом в словаре:
                if key_value and param["key"] not in ["transmission_type", "fuel_type"]:
                    current_car[param["key"]] = key_value
                elif label_value and param["key"] not in ["price"]:
                    current_car[param["key"]] = label_value
                else:
                    # Если цена в гривнах, получаем сумму, конвертированную в доллары:
                    current_car[param["key"]] = param["value"]["converted_value"] or value

        cars.append(current_car)

    return pd.DataFrame(cars)


def get_one_response(price_from: int, price_to: int):
    url = "https://www.olx.ua/api/v1/offers"

    params = {
        "offset": 0,
        "limit": 50,
        "category_id": 108,
        "owner_type": "private",
        "currency": "USD",
        "sort_by": "filter_float_price:desc",
        "filter_enum_sale_terms[0]": "regular_sale",
        "filter_refiners": "spell_checker",
        "filter_float_price:from": price_from,
        "filter_float_price:to": price_to
    }

    request = requests.Request("GET", url=url, params=params).prepare()

    cars_df = pd.DataFrame(columns=fields)

    while True:
        data_json = get_json_data(request.url)

        new_data_df = prepare_json(data_json)

        cars_df = pd.concat([cars_df, new_data_df])

        if data_json["links"].get("next"):
            request = requests.Request("GET", url=data_json["links"]["next"].get("href")).prepare()
        else:
            return cars_df


def main():
    price_range = [1, 20000]  # from-to, $
    step = 50  # $

    cars_df = pd.DataFrame(columns=fields)

    for i in range(price_range[0], price_range[1], step):
        try:
            new_cars_df = get_one_response(price_from=i, price_to=i + step)
            cars_df = pd.concat([cars_df, new_cars_df])

            print(f"${i} - ${i + step}:\t\t Найдено [{len(new_cars_df)}],\t\t всего [{len(cars_df)}]")
        except:
            break

    cars_df.dropna(how='all', inplace=True)

    cars_df.to_csv('cars.csv', index=False)

    print(f"Сохранено семплов: {len(cars_df)}.")


if __name__ == "__main__":
    main()
