import os
import io
import zipfile

import gdown

'''
Adapted from https://github.com/NVlabs/nvdiffrec
'''

def download_nerf_synthetic():
    TMP_ARCHIVE = "nerf_synthetic.zip"
    print("------------------------------------------------------------")
    print(" Downloading NeRF synthetic dataset")
    print("------------------------------------------------------------")
    nerf_synthetic_url = "https://drive.google.com/file/d/18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG/view?usp=sharing"
    gdown.download(url=nerf_synthetic_url, output=TMP_ARCHIVE, quiet=False, fuzzy=True)

    print("------------------------------------------------------------")
    print(" Extracting NeRF synthetic dataset")
    print("------------------------------------------------------------")
    archive = zipfile.ZipFile(TMP_ARCHIVE, 'r')
    for zipinfo in archive.infolist():
        if zipinfo.filename.startswith('nerf_synthetic/'):
            archive.extract(zipinfo)
    archive.close()
    os.remove(TMP_ARCHIVE)

download_nerf_synthetic()