import os
import subprocess
import sys
import tarfile

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import gdown
except ImportError:
    print("Installing gdown to handle downloads...")
    install_package("gdown")
    import gdown

# --- CONFIGURATION: File IDs from the Course Project ---
files = {
    # Annotations (Small)
    "clip_train.tar.gz": "142xxRoMaHxX3BIfCw_1b_G_dgu-02Yq3",
    
    # Training Images (3GB)
    "cc3m_subset_100k.tar.gz": "142zQjlOw0Xw4tKzXMrQjYE6NtGRTeasT",
    
    # MSCOCO Validation Images (780MB)
    "mscoco_val.tar.gz": "142tMsnclHTTPpnTXHSeNgTUlBk4She6o",
    
    # ImageNet Validation Images (6.3GB)
    "val.tar": "1NXhfhwFy-nhdABACkodgYqm9pomDKE39"
}

def setup():
    print(">>> Setting up project directories...")
    
    # 1. Create Base Folders
    os.makedirs("datasets", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    # We don't create 'clip_train' folder manually because the tarball likely creates it
    
    # 2. Create ImageNet specific folder (required for val.tar)
    os.makedirs(os.path.join("datasets", "imagenet"), exist_ok=True)

    # 3. Download Files
    print("\n>>> Downloading Data (This may take a while for big files)...")
    for filename, file_id in files.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            url = f'https://drive.google.com/uc?id={file_id}'
            # fuzz=True allows gdown to find the file even if Google warns about virus scanning for large files
            gdown.download(url, filename, quiet=False, fuzzy=True)
        else:
            print(f"{filename} already exists. Skipping download.")

    # 4. Unzipping files
    print("\n>>> Unzipping Data (This will also take some time)...")
    
    # A. Unzip Annotations (clip_train) -> Extracts to current folder
    if os.path.exists("clip_train.tar.gz"):
        print("Extracting clip_train...")
        with tarfile.open("clip_train.tar.gz", "r:gz") as tar:
            tar.extractall(path=".") # The tarball contains the folder 'clip_train'
            
    # B. Unzip CC3M Images -> Extracts into 'datasets'
    if os.path.exists("cc3m_subset_100k.tar.gz"):
        print("Extracting CC3M images...")
        with tarfile.open("cc3m_subset_100k.tar.gz", "r:gz") as tar:
            tar.extractall(path="datasets")

    # C. Unzip MSCOCO -> Extracts into 'datasets'
    if os.path.exists("mscoco_val.tar.gz"):
        print("Extracting MSCOCO images...")
        with tarfile.open("mscoco_val.tar.gz", "r:gz") as tar:
            tar.extractall(path="datasets")

    # D. Unzip ImageNet Validation -> Extracts into 'datasets/imagenet'
    if os.path.exists("val.tar"):
        print("Extracting ImageNet Val images...")
        with tarfile.open("val.tar", "r") as tar:
            tar.extractall(path=os.path.join("datasets", "imagenet"))

    print("\n✅ DONE! Your project is ready to run.")
    print("You can now delete the .tar.gz files to save space if needed.")

if __name__ == "__main__":
    setup()