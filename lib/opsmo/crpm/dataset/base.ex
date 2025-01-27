defmodule Opsmo.CRPM.Dataset.Base do
  import Nx.Defn

  def train(opts \\ []) do
    examples = Keyword.get(opts, :examples, 500)
    granularity = Keyword.get(opts, :granularity, 100)
    seed = Keyword.get(opts, :seed, 121_345)
    factor = div(examples, granularity)

    requested_seed = seed
    used_seed = seed * 2

    generate(examples: examples, granularity: granularity, factor: factor, requested_seed: requested_seed, used_seed: used_seed)
  end

  def test(opts \\ []) do
    examples = Keyword.get(opts, :examples, 50)
    granularity = Keyword.get(opts, :granularity, 10)
    seed = Keyword.get(opts, :seed, 267_434)

    requested_seed = seed
    used_seed = seed * 2

    factor = div(examples, granularity)

    generate(examples: examples, granularity: granularity, factor: factor, requested_seed: requested_seed, used_seed: used_seed)
  end

  defn generate(opts \\ []) do
    opts = keyword!(opts, examples: 500, granularity: 100, factor: 5, requested_seed: 121_345, used_seed: 267_434)
    requested_resource_key = Nx.Random.key(opts[:requested_seed])

    requested_requested =
      Nx.linspace(0.001, 1.0, n: opts[:granularity])
      |> Nx.tile([opts[:factor], 1])
      |> Nx.reshape({opts[:examples], 1})

    {requested_resource, _} = Nx.Random.shuffle(requested_resource_key, requested_requested)

    used_resource_key = Nx.Random.key(opts[:used_seed])

    used_resource =
      Nx.linspace(1.0, 0.001, n: opts[:granularity])
      |> Nx.tile([opts[:factor], 1])
      |> Nx.reshape({opts[:examples], 1})

    {used_resource, _} = Nx.Random.shuffle(used_resource_key, used_resource)

    x = Nx.concatenate([requested_resource, used_resource], axis: 1)

    y =
      x
      |> Nx.sum(axes: [1])
      |> Nx.reshape({opts[:examples], 1})
      |> Nx.greater(0.8)
      |> Nx.logical_not()
      |> Nx.as_type(:u8)
      |> Nx.equal(Nx.tensor([0, 1]))

    %{data: x, target: y}
  end
end
