""" Operations on data files e.g. merging, etc. """
import os
import pandas as pd
from sdevpy.tools import filemanager
from sdevpy import settings


def merge_tsv(path, shuffle=False):
    """ Merge all .tsv files into one, assuming they all have the same structure """
    merged_file = os.path.join(path, "merged.tsv")
    if os.path.exists(merged_file):
        print("removing file: " + merged_file)
        os.remove(merged_file)

    files = filemanager.list_files(path, [".tsv"])
    df = pd.DataFrame()
    for f in files:
        new_df = pd.read_csv(os.path.join(path, f), sep='\t')
        df = pd.concat([df, new_df])

    if shuffle:
        df = df.sample(frac=1)

    df.to_csv(merged_file, sep='\t', index=False)

if __name__ == "__main__":
    FOLDER = os.path.join(settings.WORKFOLDER, r"stovol\samples\merge")
    merge_tsv(FOLDER, shuffle=True)
