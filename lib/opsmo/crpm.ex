defmodule Opsmo.CRPM do
  @moduledoc """
  Logistic Regression model to Compute Resource Placement in a Cluster.


  ## Input

  It takes an input with the following

  - Requested CPU
  - Requested Memory
  - Requested Disk
  - Available CPU
  - Available Memory
  - Available Disk

  [requested_cpu, requested_memory, requested_disk, available_cpu, available_memory, available_disk]

  Each input value should be the normalized value between 0 and 1.

  ```elixir
  [0.05, 0.0625, 0.004, 0.825, 0.65, 0.55]
  [0.05, 0.0625, 0.004, 0.314, 0.55, 0.45]
  [0.05, 0.0625, 0.004, 0.954, 0.35, 0.55]
  ```

  The above represents 3 nodes in the cluster. You can also add more nodes to the input if you have them. The requested resource is the same for the 3 nodes. What this does is it will ask the model to tell you the best node to place the requsted resource.

  ## Usage

  This model computes the placement of resources in a cluster. It takes a 6-dimensional input and outputs a 2-dimensional vector.

  """

  @model_path "#{:code.priv_dir(:opsmo)}" <> "models/opsmo-crpm"

  @doc """
  Instantiate a new model.
  """
  def model do
    Axon.input("input", shape: {nil, 6})
    |> Axon.dense(2, name: "compute_placement", activation: :sigmoid)
  end

  @doc """
  Train the model.

  You can pass it a data stream or enumerable of data.

  ## How to use

  alias Opsmo.CRPM

  model = CRPM.model()

  data = Stream.repeatedly(fn ->
    {x, y}
  end)

  CRPM.train(model, data)
  """
  def train(model, data, opts \\ []) do
    state = Keyword.get(opts, :state) || Axon.ModelState.empty()
    iterations = Keyword.get(opts, :iterations, 100)
    epochs = Keyword.get(opts, :epochs, 100)

    model
    |> Axon.Loop.trainer(:binary_cross_entropy, Polaris.Optimizers.adamw(learning_rate: 0.01))
    |> Axon.Loop.run(data, state, iterations: iterations, epochs: epochs)
  end

  def predict(model, state, input) do
  end

  defdelegate dump(state, path \\ @model_path),
    to: Opsmo

  def load(path \\ @model_path) do
  end
end
