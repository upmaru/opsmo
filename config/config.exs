import Config

config :opsmo, :models_store, "models/"

config :opsmo, :mode, :inference

import_config "#{Mix.env()}.exs"
