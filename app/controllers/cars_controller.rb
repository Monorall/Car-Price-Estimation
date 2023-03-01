class CarsController < ApplicationController
  require 'open3'
  require 'json'

  def new
    @car = Car.new
    @car_brands = CarBrandsService.get_brands
  end

  def index

  end

  def create

  end

  def show

  end

  def evaluate
    # получение входных параметров из формы
    year = params[:year] #год
    brands = params[:brands] #марка
    model = params[:model] #модель
    condition = params[:condition] #состояние
    engine_volume = params[:engine_volume] #мощьность двигателя
    mileage = params[:mileage] #пробег
    fuel_type = params[:fuel_type] #тип топлива

    number_owners = params[:number_owners] #кол-во владельцев
    transmission_type = params[:transmission_type] #тип коробки передач
    motor_engine_size_litre = params[:motor_engine_size_litre] #объем двигателя
    drive_type = params[:drive_type] #тип привода
    color = params[:color] #цвет
    needs_body_repair = params[:needs_body_repair] #требует восстановления кузова
    needs_engine_repair = params[:needs_engine_repair] #требует восстановления двигателя
    needs_undercarriage_repair = params[:needs_undercarriage_repair] #требует восстановления подвески

    # выполнение скрипта python с передачей параметров
    stdout, stderr, status = Open3.capture3("python3
       #{Rails.root}/script/evaluate.py#{year} #{brands} #{model} #{condition} #{engine_volume} #{mileage} #{fuel_type}
                                       #{number_owners} #{transmission_type} #{motor_engine_size_litre} #{drive_type}
                                       #{color} #{needs_body_repair} #{needs_engine_repair} #{needs_undercarriage_repair}")

    # парсинг выходных данных
    result = stdout.strip.to_f

    # обработка ошибок
    if status.success?
      @price = result
      render :evaluate
    else
      flash[:error] = "Произошла ошибка: #{stderr}"
      redirect_to root_path
    end
  end

  def update_models
    brand = params[:brand]
    models = CarBrandsService.get_models(brand)
    render json: models
  end

end
