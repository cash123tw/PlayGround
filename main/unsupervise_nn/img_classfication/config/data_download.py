# download and extract data helper

import os
import requests
import zipfile

zip_file_name = 'geological_similarity.zip'
data_url = 'http://aws-proserve-data-science.s3.amazonaws.com/geological_similarity.zip'

def download(download_path):
    chunk_size = 1024
    stream = requests.get(data_url, stream=True)

    print('download start')

    dir = os.path.dirname(download_path)

    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(download_path,'wb') as f:
        for content in stream.iter_content(chunk_size=chunk_size):
            f.write(content)

    return os.path.join(dir,zip_file_name)

def extract_all(zip_path,to_path):
    with zipfile.ZipFile(zip_path) as zip:
        zip.extractall(to_path)

#test func work
if __name__ == '__main__':
    data_path = os.path.join('../../../../data', 'un_supervise_data')
    download_path = os.path.join(data_path, zip_file_name)

    if os.path.exists(data_path):
        os.removedirs(data_path)

    download(download_path)
    extract_all(download_path,data_path)

