defmodule Opsmo.CRPMTest do
  use ExUnit.Case

  setup do
    serving = Opsmo.CRPM.build_serving()

    name = :crpm_test

    {:ok, pid} = Nx.Serving.start_link(name: name, serving: serving)

    {:ok, serving: name}
  end

  describe "inference" do
    test "when requested ram is 512MB in a 8GB system and 2.4GB is available", %{serving: serving} do
      batch = Nx.Batch.stack([Nx.tensor([0.005, 0.0625, 0.003, 0.925, 0.4, 0.45])])

      assert %Nx.Tensor{} = result = Nx.Serving.batched_run(serving, batch)

      IO.insect(result)
    end
  end
end
