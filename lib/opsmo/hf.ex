defmodule Opsmo.HF do
  @hf_base "https://huggingface.co"
  @api_endpoint "https://huggingface.co/api/models"

  @organization "upmaru"

  @doc """
  Downloads model files from HuggingFace.

  ## Parameters

  - name: The model name on HuggingFace (e.g., "crpm")
  - opts: Keyword list of options
    - :branch - The branch to download from (default: "main")

  ## Example

      # Download from main branch
      Opsmo.HF.download!("crpm")

      # Download from specific branch
      Opsmo.HF.download!("crpm", branch: "dev")
  """
  def download!(model_name, opts \\ []) do
    model_name = String.downcase(model_name)
    branch = Keyword.get(opts, :branch, "main")
    path = "#{:code.priv_dir(:opsmo)}/models"

    model_path = Path.join(path, model_name)

    File.mkdir_p!(model_path)

    full_name = "#{@organization}/opsmo-#{model_name}"

    # Get file list from HF API
    files = list_model_files(full_name, branch)

    # Download all files
    Opsmo.TaskSupervisor
    |> Task.Supervisor.async_stream_nolink(files, __MODULE__, :download_file, [
      full_name,
      model_path,
      branch
    ])
    |> Enum.map(fn
      {:ok, %{body: body}} ->
        body.path

      {:error, reason} ->
        raise "Failed to download file: #{inspect(reason)}"
    end)
  end

  defp list_model_files(model_name, branch) do
    url = "#{@api_endpoint}/#{model_name}/revision/#{branch}"

    %{body: body} = Req.get!(url)

    body
    |> Map.get("siblings")
    |> Enum.map(&Map.get(&1, "rfilename"))
    |> Enum.filter(fn file ->
      String.ends_with?(file, ".safetensors") || file == "parameters.json"
    end)
  end

  def download_file(filename, model_name, dest_path, branch) do
    url = "#{@hf_base}/#{model_name}/resolve/#{branch}/#{filename}"
    dest = Path.join(dest_path, filename)

    Req.get!(url, into: File.stream!(dest))
  end
end
