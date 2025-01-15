import Config

if :os.type() == {:unix, :darwin} do
  config :nx,
    default_backend: {EMLX.Backend, device: :cpu}
end
