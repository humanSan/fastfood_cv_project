import os
import subprocess

def setup_repositories():
    """
    Clones the required repositories into the current directory.
    """
    repos = [
        ("https://github.com/simonzeng7108/efficientsam3.git", "efficientsam3"),
    ]
    
    for url, dir_name in repos:
        if not os.path.exists(dir_name):
            print(f"Cloning {url} into {dir_name}...")
            subprocess.run(["git", "clone", url, dir_name], check=True)
        else:
            print(f"Directory {dir_name} already exists. Skipping clone.")

if __name__ == "__main__":
    setup_repositories()
