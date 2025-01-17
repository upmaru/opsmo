defmodule Opsmo.CRPM.Dataset do
  @moduledoc """
  Dataset for training the CRPM model.
  """

  # 1GB in MB
  @min_memory 1024
  # 128GB in MB
  @max_memory 131_072

  def normalize_memory(memory) when is_number(memory) do
    memory
    |> Nx.tensor()
    |> normalize_memory()
  end

  def normalize_memory(%Nx.Tensor{} = memory) do
    memory
    |> Nx.subtract(@min_memory)
    |> Nx.divide(@max_memory - @min_memory)
  end

  @doc """
  Returns a stream of training data.

  ## Cases

  The training data represents all the cases we want the model to take into account.

  - When the requested ram almost match the available ram (consumes most of the available ram), the model should predict [1, 0]
  - When the requested resources is in the standard range and the resource is not fully consumed, the model should predict [0, 1]
  - When the requested resources is greater than the available resources, the model should predict [1, 0]
  - When the available resource is < 20% available, the model should predict [1, 0]

  More cases will result in a better model. We can also add more features to the model in the future, for example if the requested resource includes a GPU and the GPU is available return [0, 1].
  We can also bake GPU utilization into the dataset.

  This data generation is only good for solving cold start. Once we collect more data we should use real outcomes to further tain this model.
  """
  def memory do
    total_memory =
      Nx.tensor([2048, 4096, 8192, 16384, 32768, 65536])

    requested_memory =
      Nx.tensor([
        [128],
        [256],
        [512],
        [1024],
        [2048],
        [4096]
      ])

    normalized_total_memory =
      total_memory
      |> normalize_memory()
      |> Nx.reshape({6, 1})

    # Repeat normalized_total_memory to match x_requested shape
    normalized_total_repeated = Nx.tile(normalized_total_memory, [6, 1])

    used_memory_range =
      Nx.linspace(0.1, 1.0, n: 12)
      |> Nx.reshape({12, 1})

    x_requested =
      requested_memory
      |> Nx.divide(total_memory)
      |> Nx.reshape({36, 1})

    x_requested = Nx.concatenate([x_requested, normalized_total_repeated], axis: 1)

    x_requested_repeated = Nx.tile(x_requested, [12, 1])
    used_memory_repeated = Nx.tile(used_memory_range, [36, 1])

    # Create initial x with requested and used memory
    x =
      Nx.concatenate(
        [
          x_requested_repeated,
          used_memory_repeated
        ],
        axis: 1
      )

    # reorder the columns to match [requested, available, total_normalized]
    x = Nx.take(x, Nx.tensor([0, 2, 1]), axis: 1)

    key = Nx.Random.key(121345)

    {x, _} = Nx.Random.shuffle(key, x)

    # Calculate sum
    sum =
      Nx.add(
        # First column
        Nx.take(x, Nx.tensor([0]), axis: 1),
        # Second column
        Nx.take(x, Nx.tensor([1]), axis: 1)
      )

    # This threshold is more lenient to allow for more memory usage in higher memory systems
    # Since 0.8 of 2GB is not the same as 0.8 of 128GB
    threshold =
      Nx.take(x, Nx.tensor([2]), axis: 1)
      # Adjust this multiplier to control the effect
      |> Nx.multiply(0.30)
      # Base threshold of 0.8
      |> Nx.add(0.8)

    # Create expected output based on the sum
    # This is a simple way to generate synthetic data as a starting point
    # We're using a simple check to make sure the memory doesn't exeed 80% usage
    # For future training we should collect real data based on quality of service
    y =
      sum
      # Check if > 0.8
      |> Nx.greater(threshold)
      # Flip values (>0.8 becomes 0, <=0.8 becomes 1)
      |> Nx.logical_not()
      |> Nx.as_type(:u8)
      |> Nx.equal(Nx.tensor([0, 1]))

    {x, y}
  end

  # TODO: Implement CPU and Disk synthetic data generation we should also combine the
end
