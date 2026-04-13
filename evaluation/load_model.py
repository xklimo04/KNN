from huggingface_hub import snapshot_download
import os

snapshot_download(
    repo_id="Veronika02/target", 
    repo_type="dataset",       
    local_dir="./",
    local_dir_use_symlinks=False
)