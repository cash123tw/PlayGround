#K 鄰近辨別法,對異常值很敏感
# 參考 https://www.youtube.com/watch?v=xtaom__-drE&t=1051s
import os
import random
from collections import Counter
from typing import Union

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class K_neareast_neighbors():
    # k 代表判斷附近最近的幾個點,來決定此點的類型
    def __init__(self,k = 3):
        self.k = k

    def euclidean_distance(self,point,base_points:np.ndarray):
        #計算每個點與此點的距離
        return np.sqrt(np.sum(((base_points - point) ** 2),axis=1))

    #設定基礎資料,當作判定基礎
    def fit(self,X:Union[list,np.ndarray],Y:Union[list,np.ndarray]):
        self.base_X:np.ndarray = np.array(X)
        self.base_Y:np.ndarray = np.array(Y)

    def predict(self,point):
        X = self.base_X
        Y = self.base_Y

        distances = self.euclidean_distance(point,X)
        #取得距離最接近的 k 個 index 位置
        indices = np.argsort(distances)[:self.k]

        #取得這幾個最接近的 label
        #取得數量最多的那個
        label = Counter(Y[indices]).most_common(1)[0][0]

        return label

def get_data(len=10, feature=2, label_size=2):
    blobs = make_blobs(len,feature,centers=label_size,center_box=(-5,5))

    data = np.array(blobs[0])
    labels = np.array(blobs[1])

    return data,labels

def main():
    KNN = K_neareast_neighbors(k=3)

    #取得 base_data
    X,Y = get_data(label_size=4)
    #取得隨機生成的點
    random_point = np.random.uniform(-5.,5.,2)

    KNN.fit(X,Y)
    label = KNN.predict(random_point)

    ax = plt.subplot()
    ax.grid(True,color='white')
    ax.figure.set_facecolor('#121212')
    ax.set_facecolor('black')
    ax.tick_params(axis='x',color='white')
    ax.tick_params(axis='y',color='white')

    label_len = len(Y)
    label_color_index = {label:tuple([random.random() for _ in range(3)])
                         for i,label in enumerate(set(Y))}

    #將隨機生成點畫出來
    ax.scatter(random_point[0],random_point[1],color=label_color_index.__getitem__(label),s=200,marker='*')

    #將基礎點畫出來
    for i,data in enumerate(X):
        color = label_color_index.__getitem__(Y[i])
        ax.scatter(data[0],data[1],color=color,s=60)
        ax.plot([random_point[0],data[0]],[random_point[1],data[1]],color=color,linestyle='--',linewidth='1',)
        #計算距離
        distance =  np.sqrt(np.sum((random_point - data) ** 2))
        ax.annotate(np.round(distance,2),xy=(random_point[0],random_point[1]),xytext=(data[0],data[1]),color='white')

    plt.savefig(os.path.join(os.getcwd(), 'K-nearist-neighbors-accuracy.jpg'))
    plt.show()

if __name__ == '__main__':
    main()