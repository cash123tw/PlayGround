import os

import numpy as np
import pandas as pd

def load_data(data_src):
    column_list = [f'x{i}' for i in range(11)]
    data = pd.read_excel(data_src,names=column_list,header=None)
    return transfer_to_list(data)

def load_csv_data(data_src):
    datas = pd.read_csv(data_src,header=0)
    return [data.tolist() for data in datas.values]

def transfer_to_list(data:pd.DataFrame):
    return [list(data.iloc[i]) for i in range(len(data))]