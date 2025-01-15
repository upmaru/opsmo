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
  """
  def for_training do
    x =
      [
        [0.005, 0.640, 0.005, 0.30, 0.650, 0.782],
        [0.005, 0.0625, 0.005, 0.30, 0.30, 0.782],
        [0.005, 0.016, 0.005, 0.30, 0.650, 0.782],
        [0.005, 0.0074, 0.002, 1.0, 0.420, 0.822],
        [1, 0.5, 0.85, 1.0, 0.420, 0.822],
        [0.005, 0.016, 0.005, 0.30, 0.18, 0.182],
        [0.005, 0.0074, 0.002, 1.0, 0.20, 0.20]
      ]
      |> Nx.tensor(type: :f16)

    y =
      Nx.tensor(
        [
          [1, 0],
          [0, 1],
          [0, 1],
          [0, 1],
          [1, 0],
          [1, 0],
          [1, 0]
        ],
        type: :u8
      )

    Stream.repeatedly(fn ->
      {x, y}
    end)
  end
end
