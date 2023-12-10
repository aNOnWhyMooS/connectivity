"""Delete a model on huggingface-hub.
Usage:
    python3 delete_model.py <hub_token> <repo_name>

"""
import sys
from huggingface_hub import HfApi

hf_api = HfApi(token=sys.argv[1])
hf_api.delete_repo(sys.argv[2])
print(f'Successfully deleted: {sys.argv[2]}')
