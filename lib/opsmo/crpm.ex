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

  @model_name "crpm"

  @doc """
  Instantiate a new model.
  """
  def model do
    # Create three input tensors for CPU, Memory, and Disk
    input_cpu = Axon.input("cpu", shape: {nil, 2})
    input_memory = Axon.input("memory", shape: {nil, 2})
    input_disk = Axon.input("disk", shape: {nil, 2})

    # Create separate prediction paths for each resource
    cpu_prediction =
      Axon.dense(input_cpu, 2, activation: :sigmoid, name: "cpu")

    memory_prediction =
      Axon.dense(input_memory, 2, activation: :sigmoid, name: "memory")

    disk_prediction =
      Axon.dense(input_disk, 2, activation: :sigmoid, name: "disk")

    # Combine outputs into a single model with multiple outputs
    Axon.container(
      %{
        cpu: cpu_prediction,
        memory: memory_prediction,
        disk: disk_prediction
      },
      name: "results"
    )
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

  CRPM.train(data)
  """
  def train(data, opts \\ []) do
    save? = Keyword.get(opts, :save, false)
    model = model()

    state = Keyword.get(opts, :state) || Axon.ModelState.empty()
    iterations = Keyword.get(opts, :iterations, 100)
    epochs = Keyword.get(opts, :epochs, 100)

    # Create loss function for each output
    losses = %{
      cpu: :binary_cross_entropy,
      memory: :binary_cross_entropy,
      disk: :binary_cross_entropy
    }

    state =
      model
      |> Axon.Loop.trainer(losses, Polaris.Optimizers.adamw(learning_rate: 0.01))
      |> Axon.Loop.run(data, state, iterations: iterations, epochs: epochs)

    if save? do
      dump_state(state)
    end

    state
  end

  @doc """
  Predicts the placement score for given input resources.

  ## Parameters

  - model: The Axon model to use for prediction
  - state: The trained model state containing weights and biases
  - input: A list of 6 numbers representing [requested_cpu, requested_memory, requested_disk, available_cpu, available_memory, available_disk]

  ## Returns

  A 2-element list containing the prediction scores.

  ## Example

      iex> model = CRPM.model()
      iex> state = CRPM.load()
      iex> CRPM.predict(model, state, [0.05, 0.0625, 0.004, 0.825, 0.65, 0.55])
      [0.8234, 0.1766]

  """
  def predict(model, state, inputs) do
    # Expect inputs in the format:
    # %{
    #   "cpu" => [requested, available, sum],
    #   "memory" => [requested, available, sum],
    #   "disk" => [requested, available, sum]
    # }
    input_map = %{
      "cpu" => Nx.tensor([inputs["cpu"]]),
      "memory" => Nx.tensor([inputs["memory"]]),
      "disk" => Nx.tensor([inputs["disk"]])
    }

    Axon.predict(model, state, input_map)
  end

  def build_serving(trained_state \\ nil, batch_size \\ 3) do
    Nx.Serving.new(
      fn _options ->
        model = model()
        state = trained_state || load_state()

        template = %{
          "cpu" => Nx.template({1, 2}, :f32),
          "memory" => Nx.tempate({1, 2}, :f32),
          "disk" => Nx.template({1, 2}, :f32)
        }

        {_init_fn, predict_fn} = Axon.compile(model, template, state)

        fn inputs ->
          predict_fn.(state, inputs)
        end
      end,
      batch_size: batch_size
    )
  end

  defdelegate dump_state(state, name \\ @model_name),
    to: Opsmo,
    as: :dump

  defdelegate load_state(name \\ @model_name),
    to: Opsmo,
    as: :load
end
