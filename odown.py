#!/usr/bin/env python3
import sys
import os
import requests
from urllib.parse import urlparse
from pathlib import Path 
from tqdm import *
from zipfile import ZipFile

def main(save_path, input_link):

    url = input_link.split("?")
    url = url[0] + "?download=1"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with requests.get(url, allow_redirects=True, stream=True) as r:
        r.raise_for_status()
        filepath = urlparse(r.url)
        print("Downloading " + Path(filepath.path).name)
        with open(save_path + Path(filepath.path).name, 'wb') as f:
            pbar = tqdm(total=int(r.headers['Content-Length']))
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  
                    f.write(chunk)
                    pbar.update(len(chunk))
    with ZipFile(save_path + Path(filepath.path).name, 'r') as zipObj:
        zipObj.extractall(save_path)
    os.remove(save_path + Path(filepath.path).name)
   
if __name__ == "__main__":
    save_path = '../Features/'
    input_link = 'https://ctipub-my.sharepoint.com/:u:/g/personal/robert_bencze_stud_etti_upb_ro/EeaHosqF8btHlO6ySFhUBrMBIhLNxzR_q1O5Qd21Hf00mg?e=h3RaJZ'
    main(save_path, input_link)
