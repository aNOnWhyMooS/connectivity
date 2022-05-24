import sys

from huggingface_hub import HfApi, hf_hub_download
hf_api = HfApi()

if __name__=="__main__":
    for model in hf_api.list_models(author="connectivity"):
        for filename in hf_api.list_repo_files(model.modelId):
            if filename.endswith(".json")  and "eval" in filename:
                hf_hub_download(repo_id=model.modelId, filename=filename,
                                cache_dir=sys.argv[1], force_filename=filename)