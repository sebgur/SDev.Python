""" File management utilities """
import os
import csv
import pathlib
from io import BytesIO
import zipfile as zf
import requests


def download_unzip(zip_url, extract_folder, save_file=False):
    """ Download zip file from url and unzip """
    req = requests.get(zip_url, timeout=10)

    if save_file:
        down_filename = zip_url.split('/')[-1]
        with open(down_filename,'wb') as output_file:
            output_file.write(req.content)

    with zf.ZipFile(BytesIO(req.content)) as zip_file:
        zip_file.extractall(extract_folder)

    # zipfile = zf.ZipFile(BytesIO(req.content))
    # zipfile.extractall(extract_folder)


def check_directory(path):
    """ Creates directory if it does not already exist """
    if not os.path.exists(path):
        os.makedirs(path)


def write_csv(file):
    """ Write content to csv file """
    with open(file, mode='w', newline='', encoding='utf8') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        row = ['alpha', 'beta']
        writer.writerow(row)


def list_files(path, extensions=None):
    """ List all files in a path that have the extensions """
    all_files = os.listdir(path)
    if extensions is None:
        return all_files
    else:
        files = []
        for f in all_files:
            if pathlib.Path(f).suffix in extensions:
                files.append(f)

        return files
