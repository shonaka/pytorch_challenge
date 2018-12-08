import urllib.request
from utils.logfun import set_logger, timer
from pathlib import Path
import os
import pdb
import zipfile


def download_data(data_dir, data_name, zip_name, url):
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

    def unzip(data_dir):
        print("Unzipping the .zip file...")
        with zipfile.ZipFile(str(data_dir / zip_name), 'r') as zip_ref:
            zip_ref.extractall(str(data_dir))

def check_dir_and_create(dir_path):
    if os.path.exists(dir_path):
        print("The directory already exists.")
    else:
        print("The directory is missing, creating one")
        os.mkdir(dir_path)


if __name__ == '__main__':
    # Specifying some paths
    DATA_DIR = Path("data")
    RESULTS_DIR = Path("results")
    # Just checking if the directory exists, if not creating
    check_dir_and_create(str(DATA_DIR))
    check_dir_and_create(str(RESULTS_DIR))

    # Data URL
    URL = "https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip"
    DATA_NAME = "flower_data"
    ZIP_NAME = "flower_data.zip"

    # Custom function for logging
    log = set_logger(str(RESULTS_DIR), "pytorch_challenge.py.log")

    # Using a custom function to download the data
    download_data(data_dir=DATA_DIR, data_name=DATA_NAME, zip_name=ZIP_NAME, url=URL)