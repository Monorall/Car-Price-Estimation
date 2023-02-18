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
      render :result
    else
      render :new
    end
  end

  def evaluate
    # получение входных параметров из формы
    year = params[:year]
    brand = params[:brand]
    model = params[:model]
    condition = params[:condition]
    engine_volume = params[:engine_volume]
    mileage = params[:mileage]
    fuel_type = params[:fuel_type]

    # выполнение скрипта python с передачей параметров
    stdout, stderr, status = Open3.capture3("python3
       #{Rails.root}/script/evaluate.py#{year} #{brand} #{model} #{condition} #{engine_volume} #{mileage} #{fuel_type}")

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
