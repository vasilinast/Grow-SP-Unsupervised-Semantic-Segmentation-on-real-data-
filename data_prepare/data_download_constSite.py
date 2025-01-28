import os
import requests
import zipfile
import argparse

def download_and_extract(url, workspace_dir):
    # Ensure the workspace directory exists
    os.makedirs(workspace_dir, exist_ok=True)
    print('destination directory: ', workspace_dir)

    # Define paths
    zip_file_path = os.path.join(workspace_dir, "downloaded_folder.zip")

    # Download the file
    print("Downloading the folder...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(zip_file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Folder downloaded successfully to {zip_file_path}")
    else:
        print("Failed to download the folder. Please check the URL.")
        return

    # Extract the zip file
    print("Extracting the folder...")
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(workspace_dir)
        print(f"Folder extracted successfully to {workspace_dir}")
    except zipfile.BadZipFile:
        print("The downloaded file is not a valid zip file.")
    finally:
        # Clean up by removing the zip file
        os.remove(zip_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract a folder.")
    parser.add_argument("--url", type=str, nargs='?', default="https://tubcloud.tu-berlin.de/s/AZM2c8A4yXanNmt/download", help="URL of the folder to download (default: https://tubcloud.tu-berlin.de/s/AZM2c8A4yXanNmt/download)")
    parser.add_argument("--workspace_dir",nargs='?',default='raw_data_const_site', type=str, help="Directory to extract the folder")
    
    args = parser.parse_args()
    
    download_and_extract(args.url, args.workspace_dir)
