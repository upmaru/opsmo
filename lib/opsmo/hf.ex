defmodule Opsmo.HF do
  @hf_api_url "https://huggingface.co"
  @api_endpoint "https://huggingface.co/api/models"

  @doc """
  Downloads model files from HuggingFace.

  ## Parameters

  - name: The model name on HuggingFace (e.g., "username/model-name")

  ## Example

      Opsmo.HF.download("opsmo/crpm-v1")
  """
  def download(name) do
    [_org, model_name] = String.split(name, "/")

    path = "tmp/models"

    model_path = Path.join(path, model_name)

    File.mkdir_p!(model_path)

    # Get file list from HF API
    files = list_model_files(name)

    # Download all files
    Enum.each(files, fn file ->
      download_file(name, file, model_path)
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

  defp download_file(model_name, filename, dest_path) do
    url = "#{@hf_api_url}/#{model_name}/resolve/main/#{filename}"
    dest = Path.join(dest_path, filename)

    Req.get!(url, into: File.stream!(dest))
  end
end
