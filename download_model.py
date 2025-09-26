import os
from huggingface_hub import snapshot_download
if __name__ == '__main__':
    snapshot_download(repo_id="echo840/LIRA", local_dir='./', local_dir_use_symlinks=False, resume_download=True)
