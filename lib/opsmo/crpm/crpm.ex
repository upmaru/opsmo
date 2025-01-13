defmodule Opsmo.CRPM do
  @moduledoc """
  Compute Resource Placement Model
  """

  def model do
    Axon.input("input", shape: {nil, 6})
    |> Axon.dense(2, name: "compute_placement", activation: :sigmoid)
  end

  def train(model, data, opts \\ []) do
    state = Keyword.get(opts, :state) || Axon.ModelState.empty()
    iterations = Keyword.get(opts, :iterations, 100)
    epochs = Keyword.get(opts, :epochs, 100)

    model
    |> Axon.Loop.trainer(:binary_cross_entropy, Polaris.Optimizers.adamw(learning_rate: 0.01))
    |> Axon.Loop.run(data, state, iterations: iterations, epochs: epochs)
  end
end
