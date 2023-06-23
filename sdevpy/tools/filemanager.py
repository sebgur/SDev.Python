""" File management utilities """
import os
import csv
# import winsound
import datetime as dt
import pathlib


def check_directory(path):
    """ Creates directory if it does not already exist """
    path_exists = os.path.exists(path)
    if not path_exists:
        os.makedirs(path)


def write_csv(file):
    """ Write content to csv file """
    with open(file, mode='w', newline='', encoding='utf8') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        row = ['alpha', 'beta']
        writer.writerow(row)


# def list_csv(path):
#     """ List all csv files in a folder """
#     files = []
#     for r, d, f in os.walk(path):
#         for file in f:
#             if '.csv' in file:
#                 files.append(os.path.join(r, file))

#     return files

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

# def make_a_noise(beep=True):
#     """ Make a noise """
#     if beep:
#         f1 = 500
#         f2 = 1000
#         duration = 300
#         winsound.Beep(f1, duration)
#         winsound.Beep(f2, duration)
#         winsound.Beep(f1, duration)

#     now = dt.datetime.now()
#     dt_string = now.strftime("%H:%M:%S, %d/%m/%Y")
#     print("Closing at ", dt_string)
