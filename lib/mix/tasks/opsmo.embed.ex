defmodule Mix.Tasks.Opsmo.Embed do
  use Mix.Task

  alias Opsmo.HF

  @shortdoc "Downloads model files from HuggingFace"

  @moduledoc """
  Downloads model files from HuggingFace.

  ## Usage

      $ mix opsmo.embed MODEL_NAME[:VERSION] [MODEL_NAME[:VERSION]...]

  If no models are specified, it will use the configuration from config.exs:

      config :opsmo, :models, %{
        "crpm" => "0.3.7",
        "llm" => "main"
      }

  Examples:
      $ mix opsmo.embed crpm
      $ mix opsmo.embed crpm:v1.0.0
      $ mix opsmo.embed crpm:main llm:dev classifier
  """

  @impl Mix.Task
  def run([]) do
    Application.get_env(:opsmo, :models, %{})
    |> validate_models_config()
    |> case do
      {:ok, models} when models == %{} ->
        Mix.raise("""
        No model names provided and no models configured.

        Either provide models as arguments:
            mix opsmo.embed MODEL_NAME[:VERSION] [MODEL_NAME[:VERSION]...]

        Or configure them in config.exs:
            config :opsmo, :models, %{
              "crpm" => "0.3.7",
              "llm" => "main"
            }
        """)

      {:ok, models} ->
        System.put_env("MIX_TASK", "opsmo.embed")

        Mix.Task.run("app.start")

        models
        |> Enum.each(fn {model_name, branch} ->
          HF.download!(model_name, branch: branch)
        end)

      {:error, reason} ->
        Mix.raise("""
        Invalid models configuration:
        #{reason}

        Expected format in config.exs:
            config :opsmo, :models, %{
              "crpm" => "0.3.7",
              "llm" => "main"
            }
        """)
    end
  end

  def run(model_specs) do
    System.put_env("MIX_TASK", "opsmo.embed")

    Mix.Task.run("app.start")

    model_specs
    |> Enum.each(fn spec ->
      {model_name, branch} = parse_model_spec(spec)
      IO.puts("\nDownloading #{model_name} (#{branch})")
      HF.download!(model_name, branch: branch)
    end)
  end

  defp parse_model_spec(spec) do
    case String.split(spec, ":", parts: 2) do
      [model_name, branch] -> {model_name, branch}
      [model_name] -> {model_name, "main"}
    end
  end

  defp validate_models_config(config) when not is_map(config) do
    {:error, "Configuration must be a map, got: #{inspect(config)}"}
  end

  defp validate_models_config(config) do
    invalid_entries =
      config
      |> Enum.reject(fn
        {model, branch} when is_binary(model) and is_binary(branch) -> true
        _ -> false
      end)

    case invalid_entries do
      [] -> {:ok, config}
      entries -> {:error, "Invalid entries: #{inspect(entries)}"}
    end
  end
end
