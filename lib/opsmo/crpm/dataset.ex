defmodule Opsmo.CRPM.Dataset do
  @moduledoc """
  Dataset for training the CRPM model.
  """

  alias __MODULE__.Memory

  defdelegate memory_train, to: Memory, as: :train

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
