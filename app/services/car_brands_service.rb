class CarBrandsService

  FILE_PATH = "#{Rails.root}/config/car_data.json"

  def self.get_brands
    class << self
      brands = JSON.parse(File.read(FILE_PATH))
      brands.keys
    end
  end

  def self.get_models(brand)
    brands = JSON.parse(File.read(FILE_PATH))
    brands[brand]
  end

  def self.load_brands_and_models
    file_path = Rails.root.join('config', 'car_data.json')
    file = File.read(file_path)
    data = JSON.parse(file)

    brands = data['brands']
    models = {}

    brands.each do |brand, models_list|
      models[brand] = models_list.sort
    end

    { brands: brands, models: models }
  end
end