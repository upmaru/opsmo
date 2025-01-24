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

  @spec memory_to_parquet({any(), any()}) ::
          :ok
          | {:error, %{:__exception__ => true, :__struct__ => atom(), optional(atom()) => any()}}
  def memory_to_parquet({x, y}) do
    requested = Nx.take(x, Nx.tensor([0]), axis: 1)
    available = Nx.take(x, Nx.tensor([1]), axis: 1)
    total_normalized = Nx.take(x, Nx.tensor([2]), axis: 1)
    expected = Nx.take(y, Nx.tensor([1]), axis: 1)

    Explorer.DataFrame.new(%{
      requested: Explorer.Series.from_tensor(requested),
      used: Explorer.Series.from_tensor(available),
      total_normalized: Explorer.Series.from_tensor(total_normalized),
      expected: Explorer.Series.from_tensor(expected)
    })
    |> Explorer.DataFrame.to_parquet("datasets/memory.parquet")
  end

  # TODO: Implement CPU and Disk synthetic data generation we should also combine the
end
