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
    year = params[:year]
    cleared_customs = params[:cleared_customs]
    brand = params[:brand]
    model = params[:model]
    car_body = params[:car_body]
    color = [:color]
    transmission_type = [:transmission_type]
    drive_type = [:drive_type]
    fuel_type = [:fuel_type]
    motor_engine_size_litre = params[:motor_engine_size_litre]
    exterior_condition = params[:exterior_condition]
    lat = params[:lat]
    lon = params[:lon]
    after_an_accident = params[:after_an_accident]
    fine_condition = params[:fine_condition]
    first_owner = params[:first_owner]
    garage_storage = params[:garage_storage]
    needs_body_repair = params[:needs_body_repair]
    needs_engine_repair = params[:needs_engine_repair]
    not_bit = params[:not_bit]
    not_colored = params[:not_colored]
    not_on_the_move = params[:not_on_the_move]
    age = params[:age]

    # выполнение скрипта python с передачей параметров
    stdout, stderr, status = Open3.capture3("python3
       #{Rails.root}/script/evaluate.py#{year} #{cleared_customs} #{brand} #{model} #{car_body}
 #{color} #{transmission_type} #{drive_type} #{fuel_type} #{motor_engine_size_litre} #{exterior_condition}
 #{lat} #{lon} #{after_an_accident} #{fine_condition} #{first_owner} #{garage_storage} #{needs_body_repair}
 #{needs_engine_repair} #{not_bit} #{not_colored} #{not_on_the_move} #{age}")

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
