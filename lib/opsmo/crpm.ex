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

  defdelegate normalize_memory(memory), to: Opsmo.CRPM.Dataset.Memory

  @doc """
  Instantiate a new model.
  """
  def model do
    # Create three input tensors for CPU, Memory, and Disk
    input_cpu = Axon.input("cpu", shape: {nil, 2})
    input_memory = Axon.input("memory", shape: {nil, 3})
    input_disk = Axon.input("disk", shape: {nil, 2})

    # Create separate prediction paths for each resource
    cpu_prediction =
      Axon.dense(input_cpu, 2, activation: :sigmoid, name: "cpu")

    memory_prediction =
      input_memory
      |> Axon.dense(8, activation: :relu, name: "memory_dense_1")
      |> Axon.dense(2, activation: :sigmoid, name: "memory")

    disk_prediction =
      Axon.dense(input_disk, 2, activation: :sigmoid, name: "disk")

    # Combine outputs into a single model with multiple outputs
    Axon.container(
      {cpu_prediction, memory_prediction, disk_prediction},
      name: "results"
    )
  end

  @doc """
  Train the model.

  You can pass it a data stream or enumerable of data.

  ## How to use

  alias Opsmo.CRPM

  data = CRPM.Dataset.train()

  CRPM.train(data)
  """
  def train(data, opts \\ []) do
    compiler = Application.get_env(:opsmo, :compiler)
    save? = Keyword.get(opts, :save, false)
    model = model()

    state = Keyword.get(opts, :state) || Axon.ModelState.empty()
    iterations = Keyword.get(opts, :iterations, 100)
    epochs = Keyword.get(opts, :epochs, 100)

    # Losses and weights for each output cpu, memory, disk
    losses = [binary_cross_entropy: 0.2, binary_cross_entropy: 0.4, binary_cross_entropy: 0.4]

    state =
      model
      |> Axon.Loop.trainer(losses, Polaris.Optimizers.adamw(learning_rate: 0.01))
      |> Axon.Loop.run(data, state, iterations: iterations, epochs: epochs, compiler: compiler)

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
  def predict(state, inputs) do
    model = model()

    input_map = process_inputs(inputs)

    model
    |> Axon.predict(state, input_map)
    |> result()
  end

  def build_serving(opts \\ []) do
    trained_state = Keyword.get(opts, :trained_state, nil)
    batch_size = Keyword.get(opts, :batch_size, 3)
    autoload_state = Keyword.get(opts, :autoload_state, false)

    defn_options =
      if compiler = Application.get_env(:opsmo, :compiler) do
        [compiler: compiler]
      else
        []
      end

    # If we are in a mix task, we don't need to load the model
    if System.get_env("MIX_TASK") == "opsmo.embed" do
      Nx.Serving.new(fn _options ->
        :ok
      end)
    else
      Nx.Serving.new(
        fn _options ->
          model = model()
          state = trained_state || load_state(@model_name, autoload_state: autoload_state)

          {_init_fn, predict_fn} = Axon.build(model)

          template = %{
            "cpu" => Nx.template({batch_size, 2}, :f32),
            "disk" => Nx.template({batch_size, 2}, :f32),
            "memory" => Nx.template({batch_size, 3}, :f32)
          }

          template_args = [Nx.to_template(state), template]

          predict_fn = Nx.Defn.compile(predict_fn, template_args, defn_options)

          fn inputs ->
            inputs = Nx.Batch.pad(inputs, batch_size - inputs.size)
            predict_fn.(state, inputs)
          end
        end,
        batch_size: batch_size
      )
      |> Nx.Serving.client_postprocessing(&result/2)
    end
  end

  def process_inputs(inputs) do
    %{
      "cpu" => Nx.tensor(inputs["cpu"]),
      "memory" => Nx.tensor(inputs["memory"]),
      "disk" => Nx.tensor(inputs["disk"])
    }
  end

  defp result({outputs, _}, _), do: result(outputs)

  defp result({cpu, memory, disk}) do
    # Convert tensors to lists and take second value (probability of true) from each pair
    cpu_values = cpu |> Nx.to_list() |> Enum.map(&List.last/1)
    memory_values = memory |> Nx.to_list() |> Enum.map(&List.last/1)
    disk_values = disk |> Nx.to_list() |> Enum.map(&List.last/1)

    # Zip the values together by index
    [cpu_values, memory_values, disk_values]
    |> Enum.zip_with(fn [cpu, memory, disk] ->
      %{cpu: cpu, memory: memory, disk: disk}
    end)
  end

  defdelegate dump_state(state, name \\ @model_name),
    to: Opsmo,
    as: :dump

  defdelegate load_state(name \\ @model_name, opts \\ []),
    to: Opsmo,
    as: :load
end
