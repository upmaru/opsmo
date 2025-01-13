defmodule OpsmoTest do
  use ExUnit.Case
  doctest Opsmo

  test "greets the world" do
    assert Opsmo.hello() == :world
  end
end
