defmodule Opsmo.MixProject do
  use Mix.Project

  def project do
    [
      app: :opsmo,
      version: "0.3.8",
      elixir: "~> 1.15",
      description: description(),
      license: "Apache-2.0",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      package: package()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {Opsmo.Application, []}
    ]
  end

  defp description do
    """
    Opsmo is a collection of ML models for DevOps / SRE / Observability
    """
  end

  defp package do
    [
      name: "opsmo",
      files: ["lib", "mix.exs", "README*", "LICENSE"],
      licenses: ["Apache-2.0"],
      maintainers: ["Zack Siri"],
      links: %{"GitHub" => "https://github.com/upmaru/opsmo"}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    ci? = System.get_env("CI") == "true"

    ci = if ci?, do: :ci

    [
      {:nx, "~> 0.9"},
      {:axon, "~> 0.7"},
      {:safetensors, "~> 0.1"},
      {:req, "~> 0.5.0"},

      # Docs
      {:ex_doc, ">= 0.0.0", only: :dev, runtime: false}
      # {:dep_from_hexpm, "~> 0.3.0"},
      # {:dep_from_git, git: "https://github.com/elixir-lang/my_dep.git", tag: "0.1.0"}
    ]
    |> Enum.concat(accelerators(:os.type(), ci))
  end

  defp accelerators(_, :ci), do: []

  defp accelerators({:unix, :darwin}, _) do
    [
      {:emlx, github: "elixir-nx/emlx", only: [:dev, :test]}
    ]
  end

  defp accelerators({:unix, :linux}, _) do
    case gnu_or_musl() do
      :musl ->
        []

      :gnu ->
        [{:exla, "~> 0.9", only: [:dev, :test]}]
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
