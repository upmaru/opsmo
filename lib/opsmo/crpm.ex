defmodule Opsmo.CRPM do
  @moduledoc """
  Compute Resource Placement Model
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

  defdelegate dump(state, path \\ @model_path),
    to: Opsmo

  def load(path \\ @model_path) do
  end
end
