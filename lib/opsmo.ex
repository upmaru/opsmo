defmodule Opsmo do
  @moduledoc """
  Documentation for `Opsmo`.
  """

  @doc """
  Dump model into safetensors.
  """
  def dump(%Axon.ModelState{data: data}, path) do
    Map.keys(data)
    |> Enum.each(fn key ->
      Safetensors.write!(path, data[key])
    end)
  end
end
