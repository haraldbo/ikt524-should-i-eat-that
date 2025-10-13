"""
The following script is used to:
1. Download the Nutrition5k dataset metadata CSV files using HTTP requests.
2. Avoid the need for Google Cloud SDK (gsutil).
3. Provide a progress bar for downloads.
4. List the contents of the downloaded dataset.
This is useful for setting up the dataset for training a nutrition prediction model.
It is recommended to start by downloading only the metadata files, as the full dataset is large.

Usage:
  python models/download_nutrition5k.py --metadata-only
  This will get you the cafe 1 and cafe 2 dish metadata CSVs.
  If you dont get the ingredient metadata CSV, you can download it later directly from the source 
  @https://console.cloud.google.com/storage/browser/nutrition5k_dataset.
  nutrition5k_dataset/metadata/ingredients_metadata.csv
"""


import os
import argparse
from pathlib import Path
import requests
from tqdm import tqdm


# == 1. Helper function to download files with progress bar == #
def download_file_http(url, destination):
    """Download file via HTTP with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    # Ensure destination directory exists
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Download with progress bar
    with open(destination, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination.name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

# == 2. Main downloader class == #
class Nutrition5kDownloader:
    """Download Nutrition5k metadata via HTTP"""
    #  Default directory to store dataset
    DEFAULT_DATA_DIR = Path("./data/nutrition5k")

    #  Initialize with data directory
    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir) if data_dir else self.DEFAULT_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    #  Download metadata CSV files
    def download_metadata(self):
        """Download metadata CSVs using HTTP (no gsutil needed)"""
        print("Downloading Nutrition5k metadata via HTTP...")
        #  Base URL for metadata files
        base_http_url = "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/metadata"
        metadata_files = [
            "dish_metadata_cafe1.csv",
            "dish_metadata_cafe2.csv",
            "nutrition5k_dataset_metadata_ingredients_metadata.csv" 
        ]

        # Create metadata directory
        metadata_path = self.data_dir / "metadata"
        metadata_path.mkdir(parents=True, exist_ok=True)
        # Download each file
        for filename in metadata_files:
            url = f"{base_http_url}/{filename}"
            dest = metadata_path / filename
            print(f"⬇Downloading {filename}")
            download_file_http(url, dest)
        
        print("Metadata downloaded successfully.")
    
    # List contents of the data directory
    def list_contents(self):
        """List what's been downloaded"""
        print(f"\n Contents of {self.data_dir}:")
        if not self.data_dir.exists():
            print("  (empty - nothing downloaded yet)")
            return
        # Recursively list files
        for item in self.data_dir.rglob('*'):
            if item.is_file():
                size = item.stat().st_size / 1024 / 1024  # MB
                print(f"  {item.relative_to(self.data_dir)} ({size:.2f} MB)")

# == 3. Main execution == #
def main():
    parser = argparse.ArgumentParser(description='Download Nutrition5k metadata')
    parser.add_argument('--metadata-only', action='store_true', help='Download only metadata CSVs (recommended)')
    parser.add_argument('--data-dir', type=str, help='Directory to store dataset')
    parser.add_argument('--list', action='store_true', help='List downloaded contents')

    # Parse arguments
    args = parser.parse_args()
    downloader = Nutrition5kDownloader(data_dir=args.data_dir)

    # Handle actions based on arguments
    if args.list:
        downloader.list_contents()
        return

    # Download metadata if specified
    if args.metadata_only:
        downloader.download_metadata()
    else:
        print("Please specify --metadata-only")
        print("\nRecommended: Start with metadata only")
        print("  python models/download_nutrition5k.py --metadata-only")

if __name__ == "__main__":
    main()
