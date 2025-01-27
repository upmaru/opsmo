defmodule Mix.Tasks.Opsmo.Embed do
  use Mix.Task

  alias Opsmo.HF

  @shortdoc "Downloads model files from HuggingFace"

  @moduledoc """
  Downloads model files from HuggingFace.

  ## Usage

      $ mix opsmo.embed MODEL_NAME [MODEL_NAME...] [--branch BRANCH_NAME]

  Examples:
      $ mix opsmo.embed crpm
      $ mix opsmo.embed crpm --branch dev
      $ mix opsmo.embed crpm llm classifier
  """

  @impl Mix.Task
  def run(args) do
    {opts, model_names} = OptionParser.parse!(args, strict: [branch: :string])

    if model_names == [] do
      Mix.raise("""
      No model names provided.

      Usage:
          mix opsmo.embed MODEL_NAME [MODEL_NAME...] [--branch BRANCH_NAME]
      """)
    end

    Mix.Task.run("app.start")

    model_names
    |> Enum.each(fn model_name ->
      IO.puts("\nDownloading model: #{model_name}")

      HF.download!(model_name, branch: opts[:branch])

      IO.puts("✓ #{model_name} downloaded successfully")
    end)
  end
end
