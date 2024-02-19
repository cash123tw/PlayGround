import os

import kaggle

def download_kaggle_file(file_name, file_path, unzip=True):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    kaggle.api.authenticate()

    kaggle.api.dataset_download_files(file_name,path=file_path,unzip=unzip,quiet=False)


#test work
if __name__ == '__main__':
    download_kaggle_file(
        "tongpython/cat-and-dog",
        file_path='../../data/dog_cat',
    )
