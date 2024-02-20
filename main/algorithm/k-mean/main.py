# K-mean cluster 參考至 https://www.youtube.com/watch?v=5w5iUbTlpMQ
import os.path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

class K_clustering():
    def __init__(self,k=3):
        self.k = k
        self.centroids = None

    @staticmethod
    def distance(X,centroids):
        #計算 X 中每筆資料對 centroid 的距離
        return np.sqrt(np.sum((centroids - X) ** 2,axis=1))

    def fit(self, X, max_iteration = 200):
        X = np.array(X)
        input_shape = X.shape
        #隨機初始化中心位置
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0),
                                           size = (self.k, input_shape[1]))

        for i in range(max_iteration):
            print(f'round {i}\r')

            #每一筆算出離最近的點 index
            y = []

            #計算距離
            for row_data in X:
                distance = K_clustering.distance(row_data,self.centroids)
                y.append(np.argmin(distance))

            new_centroids = []
            y = np.array(y)
            cluster_indices = []

            for i in range(self.k):
                cluster_indices.append(np.where(y == i)[0])

            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    #若沒有資料須更新沿用舊資料
                    new_centroids.append(self.centroids[i])
                else:
                    #將該中心最近的那些資料位置,加起來取平均
                    new_centroids.append(np.mean(X[indices],axis=0))

            if np.max(self.centroids - np.array(new_centroids)) < 0.0001:
                break
            else:
                self.centroids = np.array(new_centroids)

        return y

def transfer_index(list):
    ar = []

    for i in list:
        if i not in ar:
            ar.append(i)

    return np.array([ar.index(i) for i in list])

if __name__ == '__main__':
    # data_point = np.random.uniform(0,100,size=(100,2))
    blob = make_blobs(100,2,centers=4)
    data_point = np.array(blob[0])
    k = K_clustering()
    Y = k.fit(data_point)

    #確認預測結果是否跟 make_blobs 給出的結果相同
    ari = adjusted_rand_score(blob[1],Y)

    t_Y = []
    t_blob =[]

    #準確率
    print(ari)
    #轉換 INDEX 成同樣狀態，比較看結果
    # print(transfer_index(blob[1]) == transfer_index(Y))

    plt.title(f'accuracy {ari*100:.2f}')
    plt.scatter(data_point[:,0],data_point[:,1],c=Y)
    plt.scatter(k.centroids[:,0],k.centroids[:,1],c=range(len(k.centroids)),marker='*',s=200)
    plt.savefig(os.path.join(os.getcwd(),'K-mean-accuracy.jpg'))
    plt.show()
