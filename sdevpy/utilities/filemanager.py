""" File management utilities """
import os
import csv
import pathlib
from io import BytesIO
import zipfile as zf
import requests
import pandas as pd


def merge_tsv(path: str, shuffle: bool=False):
    """ Merge all .tsv files in a path into one, assuming they all have the same structure """
    merged_file = os.path.join(path, "merged.tsv")
    if os.path.exists(merged_file):
        print("removing file: " + merged_file)
        os.remove(merged_file)

    files = list_files(path, [".tsv"])
    df = pd.DataFrame()
    for f in files:
        new_df = pd.read_csv(os.path.join(path, f), sep='\t')
        df = pd.concat([df, new_df])

    if shuffle:
        df = df.sample(frac=1)

    df.to_csv(merged_file, sep='\t', index=False)


def download_unzip(zip_url: str, extract_folder: str, save_file: bool=False):
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


def check_directory(path: str):
    """ Creates directory if it does not already exist """
    if not os.path.exists(path):
        os.makedirs(path)


def write_csv(file: str):
    """ Write content to csv file """
    with open(file, mode='w', newline='', encoding='utf8') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        row = ['alpha', 'beta']
        writer.writerow(row)


def list_files(path: str, extensions=None):
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
