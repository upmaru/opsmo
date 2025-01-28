defmodule Opsmo.MixProject do
  use Mix.Project

  def project do
    [
      app: :opsmo,
      version: "0.3.0",
      elixir: "~> 1.15",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {Opsmo.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, "~> 0.9"},
      {:axon, "~> 0.7"},
      {:safetensors, "~> 0.1"},
      {:req, "~> 0.5.0"}
      # {:dep_from_hexpm, "~> 0.3.0"},
      # {:dep_from_git, git: "https://github.com/elixir-lang/my_dep.git", tag: "0.1.0"}
    ]
    |> Enum.concat(accelerators(:os.type()))
  end

  defp accelerators({:unix, :darwin}) do
    [
      {:emlx, github: "elixir-nx/emlx"}
    ]
  end

  defp accelerators({:unix, :linux}) do
    case gnu_or_musl() do
      :musl ->
        []

      :gnu ->
        [{:exla, "~> 0.9"}]
    end
  end

  defp gnu_or_musl do
    {output, _} = System.cmd("ldd", ["--version"], stderr_to_stdout: true)

    cond do
      String.contains?(output, "musl") ->
        :musl

      String.contains?(output, "GLIBC") ->
        :gnu
    end
  end
end
