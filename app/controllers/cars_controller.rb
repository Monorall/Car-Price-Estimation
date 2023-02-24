class CarsController < ApplicationController
  require 'open3'

  def index

  end
  def new
    @car = Car.new
  end

  def create
    @car = Car.new(evaluate)

    if @car.save
      render json: { result: @car }, status: :ok
    else
      render json: { errors: @car.errors.full_messages }, status: :unprocessable_entity
    end
  end

  def evaluate
    # получение входных параметров из формы
    year = params[:year] #год
    brand = params[:brand] #марка
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
       #{Rails.root}/script/evaluate.py#{year} #{brand} #{model} #{condition} #{engine_volume} #{mileage} #{fuel_type}
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
end
