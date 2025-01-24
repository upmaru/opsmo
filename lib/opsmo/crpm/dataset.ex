defmodule Opsmo.CRPM.Dataset do
  @moduledoc """
  Dataset for training the CRPM model.
  """

  alias __MODULE__.Memory
  alias __MODULE__.Base

  def train do
    memory = Memory.train()
    disk = Base.train()
    cpu = Base.train()

    inputs = %{
      "cpu" => cpu.data,
      "memory" => memory.data,
      "disk" => disk.data
    }

    Stream.repeatedly(fn ->
      {inputs, {cpu.target, memory.target, disk.target}}
    end)
  end

  def test(samples \\ 3) do
    memory = Memory.train()
    disk = Base.train()
    cpu = Base.train()

    memory_sample = Nx.slice_along_axis(memory.data, 0, samples, axis: 0)
    memory_target = Nx.slice_along_axis(memory.target, 0, samples, axis: 0)

    disk_sample = Nx.slice_along_axis(disk.data, 0, samples, axis: 0)
    disk_target = Nx.slice_along_axis(disk.target, 0, samples, axis: 0)

    cpu_sample = Nx.slice_along_axis(cpu.data, 0, samples, axis: 0)
    cpu_target = Nx.slice_along_axis(cpu.target, 0, samples, axis: 0)

    inputs = %{
      "cpu" => Nx.to_list(cpu_sample),
      "memory" => Nx.to_list(memory_sample),
      "disk" => Nx.to_list(disk_sample)
    }

    targets = %{
      cpu: cpu_target,
      memory: memory_target,
      disk: disk_target
    }

    {inputs, targets}
  end
end
