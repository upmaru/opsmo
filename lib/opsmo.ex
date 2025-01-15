defmodule Opsmo do
  @moduledoc """
  Documentation for `Opsmo`.
  """

  @doc """
  Dump model into safetensors.
  """
  def dump(%Axon.ModelState{data: data, parameters: parameters}, name) do
    path = Application.get_env(:opsmo, :models_path) <> "/" <> name
    File.mkdir_p!(path)

    File.write!(path <> "/parameters.json", Jason.encode_to_iodata!(parameters))

    Map.keys(data)
    |> Enum.each(fn key ->
      layer_path = path <> "/" <> key <> ".safetensors"

      Safetensors.write!(layer_path, data[key])
    end)
  end

  def load(name) do
    path = "#{:code.priv_dir(:opsmo)}/models" <> "/" <> name

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

        {layer_name, Safetensors.read!(path <> "/" <> layer)}
      end)
      |> Enum.into(%{})

    state = Axon.ModelState.empty()

    %{state | data: tensors, parameters: parameters}
  end
end
