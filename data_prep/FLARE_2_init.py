import os
from huggingface_hub import snapshot_download
import subprocess
import shutil

'''
Make sure we have 7z first:
sudo apt install p7zip-full
'''

# Download the FLARE-2 dataset for Task 2: Laptop Segmentation
local_dir = "./data/FLARE-Task2-LaptopSeg"
os.makedirs(local_dir, exist_ok=True)
snapshot_download(
    repo_id="FLARE-MedFM/FLARE-Task2-LaptopSeg",
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)

shutil.rmtree("data/FLARE-Task2-LaptopSeg/.cache")


# Unzip the specific .7z file using 7z command-line tool
file1 = "./data/FLARE-Task2-LaptopSeg/train_pseudo_label/pseudo_label_aladdin5_flare22.7z"
file2 = "./data/FLARE-Task2-LaptopSeg/train_pseudo_label/pseudo_label_blackbean_flare22.zip"
output_dir = "./data/FLARE-Task2-LaptopSeg/train_pseudo_label"
subprocess.run([
    "7z", "x", file1, f"-o{output_dir}"
], check=True)
os.remove(file1)
subprocess.run([
    "7z", "x", file2, f"-o{output_dir}"
], check=True)
os.remove(file2)

shutil.rmtree("data/FLARE-Task2-LaptopSeg/train_pseudo_label/__MACOSX")

# Recursively rename files to replace "_0000." with "."
def rename_files(directory):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if "_0000." in filename:
                new_filename = filename.replace("_0000.", ".")
                os.rename(os.path.join(root, filename), os.path.join(root, new_filename))

rename_files("./data")

for i in range(1, 10):
    os.rename(f"data/FLARE-Task2-LaptopSeg/train_pseudo_label/flare22_aladdin5_pseudo/Case{i}.nii.gz",
        f"data/FLARE-Task2-LaptopSeg/train_pseudo_label/flare22_aladdin5_pseudo/Case_0000{i}.nii.gz")