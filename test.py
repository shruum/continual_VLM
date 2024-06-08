
import os
import glob
import zipfile
from zipfile import ZipFile
from os import listdir
from os.path import isfile, join

def method1(path):
    # path = os.getcwd()
    files = glob.glob(os.path.join(path, '/**/*.zip'), recursive=True)
    index = 0

    for i in range(0, len(files)):
        with ZipFile(files[index], mode='r') as zip:
            zip.extractall()
            print("Extraction completed! " + str(files[index]) + " has been extracted!")
        index += 1
    # else:
    #     print("Not the specified directory. Go to the specified directory and it must contain this Python Script to extract all zip files.")


def method2(directory):
    onlyfiles = [f for f in listdir(directory)] #if isfile(join(directory, f))]
    for o in onlyfiles:
        if o.endswith(".zip"):
            zip = zipfile.ZipFile(os.path.join(directory, o), 'r')
            zip.extractall(directory)
            zip.close()

def method3(path):
    for root, dirnames, filenames in os.walk(path):
        for f in filenames:
            if f.endswith(".zip"):
                zip = zipfile.ZipFile(os.path.join(root, f), 'r')
                zip.extractall(root)
                zip.close()

method3('/data/users/shruthi.gowda/Pictures/folder1')