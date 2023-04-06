Rails.application.routes.draw do
  # Define your application routes per the DSL in https://guides.rubyonrails.org/routing.html

  # Defines the root path route ("/")
  # root "articles#index"

  # Defines the root path route ("/")
  root "cars#index"

  # Cars routes
  get "/cars", to: "cars#index"
  get "/cars/new", to: "cars#new"
  post "/cars", to: "cars#create"
  get "/cars/:id", to: "cars#show"
  get "/cars/:id/edit", to: "cars#edit"
  patch "/cars/:id", to: "cars#update"
  delete "/cars/:id", to: "cars#destroy"

  get "/cars/evaluate", to: "cars#evaluate"
end
