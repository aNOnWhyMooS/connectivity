import os, sys
from huggingface_hub import HfApi

hf_api = HfApi()

if __name__ == "__main__":
    repo_name, file_location, auth_token = sys.argv[1], sys.argv[2], sys.argv[3]

    if len(sys.argv) > 4:
        path_in_repo = sys.argv[4]
    else:
        path_in_repo = None

    hf_api.upload_file(
        path_or_fileobj=file_location,
        path_in_repo=path_in_repo,
        repo_id=repo_name,
        token=auth_token,
    )
