defmodule Opsmo.CRPM.Dataset.Base do
  import Nx.Defn

  def train(examples \\ 500, granularity \\ 100) do
    factor = div(examples, granularity)

    generate(examples: examples, granularity: granularity, factor: factor)
  end

  def test(examples \\ 50, granularity \\ 50) do
    factor = div(examples, granularity)

    generate(examples: examples, granularity: granularity, factor: factor)
  end

  defn generate(opts \\ []) do
    opts = keyword!(opts, examples: 500, granularity: 100, factor: 5)
    requested_resource_key = Nx.Random.key(121_345)

    requested_requested =
      Nx.linspace(0.001, 1.0, n: opts[:granularity])
      |> Nx.tile([opts[:factor], 1])
      |> Nx.reshape({opts[:examples], 1})

    {requested_resource, _} = Nx.Random.shuffle(requested_resource_key, requested_requested)

    used_resource_key = Nx.Random.key(3_455_212)

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
