defmodule Opsmo.Application do
  use Application

  def start(_type, _args) do
    children = [
      {Task.Supervisor, name: Opsmo.TaskSupervisor}
    ]

    opts = [strategy: :one_for_one, name: Opsmo.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
