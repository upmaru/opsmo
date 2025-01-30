import Config

config :opsmo, :models_store, "models/"

config :opsmo, :mode, :inference

config :opsmo, :models, %{
  "crpm" => "0.3.7"
}

import_config "#{Mix.env()}.exs"
