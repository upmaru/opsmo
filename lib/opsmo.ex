defmodule Opsmo do
  alias Opsmo.HF

  @moduledoc """
  Documentation for `Opsmo`.
  """

  @valid_models [Opsmo.CRPM]

  def start_link(model) when model in @valid_models do
    serving = model.build_serving()

    Nx.Serving.start_link(name: model, serving: serving)
  end

  def spec(model, opts \\ []) when model in @valid_models do
    {autoload_state, opts} = Keyword.pop(opts, :autoload_state, false)

    serving = model.build_serving(autoload_state: autoload_state)

    options =
      [name: model, serving: serving]
      |> Keyword.merge(opts)

    {Nx.Serving, options}
  end

  def append_model_spec(children, model, opts \\ []) do
    if System.get_env("MIX_TASK") == "opsmo.embed" do
      children
    else
      children ++ [spec(model, opts)]
    end
  end

  def predict(model, inputs) do
    inputs = model.process_inputs(inputs)

    batch = Nx.Batch.concatenate([inputs])

    Nx.Serving.batched_run(model, batch)
  end

  @doc """
  Dump model into safetensors.
  """
  def dump(%Axon.ModelState{data: data, parameters: parameters}, name) do
    path = Application.get_env(:opsmo, :models_store) <> "/" <> name

    File.mkdir_p!(path)

    File.write!(path <> "/parameters.json", Jason.encode_to_iodata!(parameters))

    Map.keys(data)
    |> Enum.each(fn key ->
      layer_path = path <> "/" <> key <> ".safetensors"

      Safetensors.write!(layer_path, data[key])
    end)
  end

  def load(name, opts \\ []) do
    autoload_state = Keyword.get(opts, :autoload_state, false)
    mode = Application.get_env(:opsmo, :mode, :inference)
    models_config = Application.get_env(:opsmo, :models, %{})

    branch = Map.get(models_config, name, "main")

    path = models_path(mode) <> String.downcase(name)

    # Check if model files exist
    has_files =
      case File.ls(path) do
        {:ok, files} ->
          has_params = Enum.member?(files, "parameters.json")
          has_tensors = Enum.any?(files, &String.ends_with?(&1, ".safetensors"))
          has_params && has_tensors

        {:error, _} ->
          false
      end

    # Download if files are missing
    cond do
      !has_files && autoload_state && mode == :inference ->
        IO.puts("Model files not found. Downloading #{name}...")
        HF.download!(name, branch: branch)

      !has_files && mode == :inference ->
        raise """
          Model files not found. Please add

            config :opsmo, :models, %{
              #{name} => \"#{branch}\"
            }

          In config.exs and run `mix opsmo.embed` to download the model.
        """

      has_files ->
        :ok
    end

    parameters =
      File.read!(path <> "/parameters.json")
      |> Jason.decode!()

    layers =
      File.ls!(path)
      |> Enum.filter(fn p ->
        String.ends_with?(p, ".safetensors")
      end)

    tensors =
      Enum.map(layers, fn layer ->
        layer_name = String.replace(layer, ".safetensors", "")

        tensor_path = path <> "/" <> layer

        {layer_name, Safetensors.read!(tensor_path)}
      end)
      |> Enum.into(%{})

    state = Axon.ModelState.empty()

    %{state | data: tensors, parameters: parameters}
  end

  defp models_path(:training) do
    Application.get_env(:opsmo, :models_store)
  end

  defp models_path(:inference) do
    "#{:code.priv_dir(:opsmo)}/models/"
  end
end
