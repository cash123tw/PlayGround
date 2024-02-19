import os.path

import tensorflow.python.client.session
import tensorflow as tf
import numpy as np

# from data_download import unpack_path
from matplotlib.image import imread
from matplotlib import pyplot as plt
from main.unsupervise_nn.img_classfication.config.data_download import *

class DataLoader():

    def __init__(self, data_src, valid_data_rate=0.1):
        self.test_data_rate = valid_data_rate
        #讀取照片資料 label_tuple_list -> (label_index:int,label_name:str)
        self.label_key_list,self.train_image_info_map,self.test_image_info_map = self.load_data_info(data_src,valid_data_rate)

    def load_data_info(self,data_path,valid_data_rate):
        label_key_list = []
        train_image_info_map = {}
        test_image_info_map = {}

        if not os.path.exists(data_path):
            dir = os.path.dirname(data_path)

            if not os.path.exists(dir): os.makedirs(dir)

            file_path = download(os.path.join(dir,zip_file_name))
            extract_all(file_path,dir)

        for dir_name in os.listdir(data_path):
            dir_path = os.path.join(data_path,dir_name)

            if not os.path.isdir(dir_path): continue

            #包裝成 {} 型態,其中img_path包含圖片路徑,需用到時候在加載,img_data 放已經加載的圖片資訊,防止重複加載
            all_img_path = [{'img_path':os.path.join(dir_path,img_name),'img_data':None} for img_name in os.listdir(dir_path)]

            img_len = len(all_img_path)
            #分割訓練資料和驗證資料
            label_key_list.append(dir_name)
            train_image_info_map.__setitem__(dir_name,np.array(all_img_path[:int(-img_len*valid_data_rate)]))
            test_image_info_map.__setitem__(dir_name,np.array(all_img_path[int(-img_len*valid_data_rate):]))

        return label_key_list,train_image_info_map,test_image_info_map

    def get_triples_data(self,test_data=False)->tuple:
        [label_l,label_r] = np.random.randint(0,len(self.label_key_list)-1,2)
        #取得 index 因為,train_image_info_map中的key是label_name
        (a,p) = self.get_img_from_info(label_l,2,test_data=test_data)
        (n,) = self.get_img_from_info(label_r,test_data=test_data)

        return a,p,n,label_r,label_r,label_l

    def get_img_from_info(self, label_index, size=1,test_data=False) -> tuple:
        data_source = self.train_image_info_map if not test_data else self.test_image_info_map

        label_name = self.label_key_list[label_index]
        img_info_list = data_source.__getitem__(label_name)

        result = [
            img_info_list[i]['img_data'] if img_info_list[i]['img_data'] else imread(img_info_list[i]['img_path'])
            for i in np.random.randint(0,len(img_info_list)-1,size)
        ]

        result = tf.image.resize(result,(32,32))

        return tuple(result)

    def get_tuple_batch(self, data_size=32, test_data=False):
        anchor_batch = []
        positive_batch = []
        negative_batch = []
        anchor_labels = []
        positive_labels = []
        negative_labels = []

        data_src_len = data_size if data_size != -1 else len(self.test_image_info_map) if test_data else len(self.train_image_info_map)

        for _ in range(data_src_len):
            a,p,n,la,lp,ln = self.get_triples_data(test_data=test_data)
            anchor_batch.append(a)
            positive_batch.append(p)
            negative_batch.append(n)
            anchor_labels.append(la)
            positive_labels.append(lp)
            negative_labels.append(ln)

        return np.array(anchor_batch),np.array(positive_batch),np.array(negative_batch),\
                np.array(anchor_labels), np.array(positive_labels), np.array(negative_labels)

    def get_tuple_batch_data_set(self, data_size, test_data=False):
        data_label_list = self.get_tuple_batch(data_size, test_data=test_data)

        data_list = [tf.data.Dataset.from_tensor_slices(data) for data in data_label_list[:3]]
        label_list = [tf.data.Dataset.from_tensor_slices(data) for data in data_label_list[3:]]

        return tensorflow.data.Dataset.zip(tuple(data_list)), \
                tensorflow.data.Dataset.zip(tuple(label_list))

    def show_img(self,target_list,label_list,round=1):
        target_list = np.array(target_list)
        label_list = np.array(label_list)

        for _ in range(round):
            index = np.random.permutation(len(target_list))
            target_list = target_list[index]
            label_list = label_list[index]

            fig, ax = plt.subplots(6, 3)

            for i, img_datas in enumerate(target_list):

                if i >= 6:
                    break

                label_datas = label_list[i]

                for c, data in enumerate(img_datas):
                    ax[i][c].imshow(data)
                    ax[i][c].set_title(label_datas[c])

            plt.show()

#查看資料正確性
# if __name__ == '__main__':
#     file_path = os.path.join(unpack_path, ('geological_similarity'))
#     dl = DataLoader(file_path)
#     x,y = dl.get_tuple_batch_data_set(50)
#     dl.show_img(list(x.as_numpy_iterator()),list(y.as_numpy_iterator()),10)

