defmodule Opsmo.CRPM.Dataset do
  @moduledoc """
  Dataset for training the CRPM model.
  """

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

    used_memory_range =
      Nx.linspace(0.1, 1.0, n: 12)
      |> Nx.reshape({12, 1})

    x_requested =
      requested_memory
      |> Nx.divide(total_memory)
      |> Nx.reshape({36, 1})

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

    # Calculate sum
    sum =
      Nx.add(
        # First column
        Nx.slice_along_axis(x, 1, 1, axis: 1),
        # Second column
        Nx.slice_along_axis(x, 0, 1, axis: 1)
      )

    # Create expected output based on the sum
    # This is a simple way to generate synthetic data as a starting point
    # We're using a simple check to make sure the memory doesn't exeed 80% usage
    # For future training we should collect real data based on quality of service
    y =
      sum
      # Check if > 0.8
      |> Nx.greater(0.8)
      # Flip values (>0.8 becomes 0, <=0.8 becomes 1)
      |> Nx.logical_not()
      |> Nx.as_type(:u8)
      |> Nx.equal(Nx.tensor([0, 1]))

    {x, y}
  end

  # TODO: Implement CPU and Disk synthetic data generation we should also combine the
end
