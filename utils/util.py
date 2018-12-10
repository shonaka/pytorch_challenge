import os
import zipfile
import urllib.request

def download_data(data_dir, data_name, zip_name, url):
    # Function to unzip
    def unzip(data_dir):
        print("Unzipping the .zip file...")
        with zipfile.ZipFile(str(data_dir / zip_name), 'r') as zip_ref:
            zip_ref.extractall(str(data_dir))

    # First check if the file exists already
    if os.path.exists(str(data_dir / zip_name)) or os.path.exists(str(data_dir / data_name)):
        print("Zip file already exists. Checking if it needs unzipping.")
        if os.path.exists(str(data_dir / data_name)):
            print("You are good to go.")
        else:
            # Only need to unzip
            unzip(data_dir)
    else: # Needs to download and then unzip
        print("Downloading the .zip file... it may take a while...")
        urllib.request.urlretrieve(url, str(data_dir / zip_name))
        # Now unzip
        unzip(data_dir)

def check_dir_and_create(dir_path):
    if os.path.exists(dir_path):
        print("The directory already exists.")
    else:
        print("The directory is missing, creating one")
        os.mkdir(dir_path)



