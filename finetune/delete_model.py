"""Delete a model on huggingface-hub.
Usage:
    python3 delete_model.py <hub_token> <repo_name|repo_substr>

Deletes repo with <repo_name> or all repos having <repo_substr> in 
their id.
"""
import sys
from huggingface_hub import HfApi, utils

hf_api = HfApi(token=sys.argv[1], endpoint="https://huggingface.co")
try:
    hf_api.delete_repo(sys.argv[2])
    print(f'Successfully deleted: {sys.argv[2]}')
except utils.RepositoryNotFoundError:
    models = [model.id for model in hf_api.list_models(search=sys.argv[2])]
    should_delete = (input(f'Delete {len(models)} models having {sys.argv[2]} ? (y/n):')=='y')
    if should_delete:
        for model in models:
            hf_api.delete_repo(model)
            print('Successfully deleted:', model)