defmodule Mix.Tasks.Opsmo.Embed do
  use Mix.Task

  @shortdoc "Downloads model files from HuggingFace"

  @moduledoc """
  Downloads model files from HuggingFace.

  ## Usage

      $ mix opsmo.embed MODEL_NAME [MODEL_NAME...]

  Examples:
      $ mix opsmo.embed crpm
      $ mix opsmo.embed crpm llm classifier
  """

  @impl Mix.Task
  def run([]) do
    Mix.raise("""
    No model names provided.

    Usage:
        mix opsmo.embed MODEL_NAME [MODEL_NAME...]
    """)
  end

  def run(model_names) do
    IO.inspect(model_names)

    Mix.Task.run("app.start")

    results =
      model_names
      |> Enum.map(fn model_name ->
        IO.puts("\nDownloading model: #{model_name}")

        case Opsmo.HF.download(model_name) do
          items when is_list(items) ->
            IO.puts("✓ #{model_name} downloaded successfully")
            {:ok, model_name}

          {:error, reason} ->
            IO.puts("✗ Failed to download #{model_name}: #{inspect(reason)}")
            {:error, model_name, reason}
        end
      end)

    case Enum.filter(results, &(elem(&1, 0) == :error)) do
      [] ->
        :ok

      failures ->
        failures
        |> Enum.map_join("\n", fn {:error, name, reason} ->
          "  #{name}: #{inspect(reason)}"
        end)
        |> then(&Mix.raise("Some models failed to download:\n#{&1}"))
    end
  end
end
