defmodule Opsmo.HF do
  @hf_api_url "https://huggingface.co"
  @api_endpoint "https://huggingface.co/api/models"

  @organization "upmaru"

  @doc """
  Downloads model files from HuggingFace.

  ## Parameters

  - name: The model name on HuggingFace (e.g., "crpm")

  ## Example

      Opsmo.HF.download("crpm")
  """
  def download(model_name) do
    model_name = String.downcase(model_name)
    path = "tmp/models"

    model_path = Path.join(path, model_name)

    File.mkdir_p!(model_path)

    full_name = "#{@organization}/opsmo-#{model_name}"

    # Get file list from HF API
    files = list_model_files(full_name)

    # Download all files
    Opsmo.TaskSupervisor
    |> Task.Supervisor.async_stream_nolink(files, __MODULE__, :download_file, [
      full_name,
      model_path
    ])
    |> Enum.map(fn
      {:ok, %{body: body}} ->
        body.path

      {:error, reason} ->
        raise "Failed to download file: #{inspect(reason)}"
    end)
  end

  defp list_model_files(model_name) do
    url = "#{@api_endpoint}/#{model_name}"

    %{body: body} = Req.get!(url)

    body
    |> Map.get("siblings")
    |> Enum.map(&Map.get(&1, "rfilename"))
    |> Enum.filter(fn file ->
      String.ends_with?(file, ".safetensors") || file == "parameters.json"
    end)
  end

  def download_file(filename, model_name, dest_path) do
    url = "#{@hf_api_url}/#{model_name}/resolve/main/#{filename}"
    dest = Path.join(dest_path, filename)

    Req.get!(url, into: File.stream!(dest))
  end
end
