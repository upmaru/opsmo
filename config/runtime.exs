import Config

if :os.type() == {:unix, :darwin} do
  config :nx,
    default_backend: {EMLX.Backend, device: :gpu}
end

if :os.type() == {:unix, :linux} do
  config :nx, :default_backend, {EXLA.Backend, client: :cuda}
end
