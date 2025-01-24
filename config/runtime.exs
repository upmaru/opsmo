import Config

if :os.type() == {:unix, :darwin} do
  config :nx,
    default_backend: {EMLX.Backend, device: :gpu}

  config :opsmo, :compiler, EMLX
end

if :os.type() == {:unix, :linux} do
  config :nx, :default_backend, {EXLA.Backend, client: :cuda}

  config :opsmo, :compiler, EXLA
end
