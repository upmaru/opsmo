defmodule Mix.Tasks.Opsmo.Embed do
  use Mix.Task

  alias Opsmo.HF

  @shortdoc "Downloads model files from HuggingFace"

  @moduledoc """
  Downloads model files from HuggingFace.

  ## Usage

      $ mix opsmo.embed MODEL_NAME[:VERSION] [MODEL_NAME[:VERSION]...]

  Examples:
      $ mix opsmo.embed crpm
      $ mix opsmo.embed crpm:v1.0.0
      $ mix opsmo.embed crpm:main llm:dev classifier
  """

  @impl Mix.Task
  def run([]) do
    Mix.raise("""
    No model names provided.

    Usage:
        mix opsmo.embed MODEL_NAME[:VERSION] [MODEL_NAME[:VERSION]...]
    """)
  end

  def run(model_specs) do
    Mix.Task.run("app.start")

    model_specs
    |> Enum.each(fn spec ->
      {model_name, branch} = parse_model_spec(spec)
      IO.puts("\nDownloading model: #{model_name} (#{branch})")

      HF.download!(model_name, branch: branch)

      IO.puts("✓ #{model_name} downloaded successfully")
    end)
  end

  defp parse_model_spec(spec) do
    case String.split(spec, ":", parts: 2) do
      [model_name, branch] -> {model_name, branch}
      [model_name] -> {model_name, "main"}
    end
  end
end
