import os
from huggingface_hub import snapshot_download
from argparse import ArgumentParser
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', '-n', type=str, default="echo840/LIRA") # OpenGVLab/InternVL2-2B  OpenGVLab/InternVL2-8B
    args = parser.parse_args()
    if "LIRA" in args.name:
        snapshot_download(repo_id=args.name, local_dir='./', local_dir_use_symlinks=False, resume_download=True)
    else:
        if not os.path.exists('./pretrained/'):
            os.makedirs('./pretrained/')
        snapshot_download(repo_id=args.name, local_dir='./pretrained/'+args.name.split('/')[-1], local_dir_use_symlinks=False, resume_download=True)