import os
import random
import statistics

import pandas as pd

from tree import Node
from tree_func import *
from data_load import *
import pickle

root_path = '../../lib'
save_path = os.path.join(root_path,'decision_tree.pickle')
save_path_2 = os.path.join(root_path,'decision_random_tree.pickle')

def get_split(dataset):
    #預設值,都取極大值
    best_index,best_threshold,best_gini,best_groups = 9999,9999,9999,None
    flag_terminal = False
    #欄位長度,不包含最後一列
    column_num = len(dataset[0][:-1])
    #取得所有類別
    classes_set = set([row[-1] for row in dataset])
    classes = list(classes_set)

    for index in range(column_num):
        # 排序
        dataset = sorted(dataset, key=itemgetter(index))
        threshold_list = get_threshold(index,dataset)

        #用每個閥值去分割資料,找出 gini 為最小值
        for threshold in threshold_list:
            groups = data_split(index,threshold,dataset)
            gini = gini_index(groups,classes)

            if gini < best_gini:
                best_gini = gini
                best_groups = groups
                best_index = index
                best_threshold = threshold

    #巡迴完成找到最優的分類法,建造節點
    node = Node(best_index,best_threshold)
    #判斷是否結束
    flag_terminal = (best_gini == 0 or data_same(dataset))

    if flag_terminal:
        node.set_label(best_groups,classes)
    else:
        node.set_left(get_split(best_groups[0]))
        node.set_right(get_split(best_groups[1]))

    return node

def prediction(data:list,node):
    err,results = 0.,[]

    index,threshold = node.index,node.threshold
    #資料排序
    data = sorted(data,key=itemgetter(index))
    #分類資料
    groups = data_split(index,threshold,data)

    #若已經沒有子節點,則將分類結果取出
    if node.left == None:
        #分類到左節點資料
        pack = wrap_prediction(groups[0],node.label_l)
        err += pack[0]
        results += pack[1]
        # 分類到右節點資料
        pack = wrap_prediction(groups[1], node.label_r)
        err += pack[0]
        results += pack[1]
    #若還有子節點則繼續分類
    else:
        err_l,result_l = prediction(groups[0],node.left)
        err_r,result_r = prediction(groups[1],node.right)

        err += (err_l + err_r)

        results += result_l
        results += result_r

    return err,results

def wrap_prediction(group,label):
    err_count,result = 0,[]

    for row in group:
        result.append((row,label))

        if label != row[-1]:
            err_count += 1

    return err_count,result

#取得隨機比例的資料
def get_random_data(data,fraction):
    data_len = int(len(data) * fraction)
    index_list = random.choices(range(0,len(data)),k=data_len)
    return [data[i] for i in index_list]

def train_random_trees(data,tree_num:int = 10,fraction=0.7):
    trees:List[Node] = []

    for i in range(tree_num):
        train_data = get_random_data(data,fraction)
        trees.append(get_split(train_data))
        print(f'train {i} complete')

    with open(save_path_2,'wb') as f:
        pickle.dump(trees,f)

def test_random_trees(data):
    with open(save_path_2,'rb') as f:
        trees:List[Node] = pickle.load(f)

    errs = []
    results = {}

    for node in trees:
        err,result = prediction(data,node)

        errs.append(err)

        #將預測結果進行投票
        for (row,label) in result:
            #判定結果存到 results 中 key 為 tuple(row)
            key = tuple(row)
            if key not in results:
                value_map = {}
                results.__setitem__(key,value_map)
            else :
                value_map = results.__getitem__(key)

            if label in value_map:
                value = value_map.get(label)
                value += 1
                value_map.__setitem__(label,value)
            else:
                value_map.__setitem__(label,1)

    random_err = 0

    #預測完後，拿出預測值最高的結果,並計算錯誤率
    for key in results.keys():
        true_label = key[-1]
        pred_label = max(results[key],key=lambda k:results[key][k])

        random_err += (1 if (true_label != pred_label) else 0)

    print(f'mean err rate {statistics.mean(errs)/len(data)*100}% ,random err rate {random_err/len(data)*100}%')

def train(data):
    node = get_split(data)
    with open(save_path, 'wb') as f:
        pickle.dump(node, f)

def test(data):
    with open(save_path,'rb') as f:
        node:Node = pickle.load(f)

    err,results = prediction(data,node)
    print(f'{err / len(data)*100}%')

iris_data = os.path.join("../../data/iris",'Iris.csv')

def load_iris_data(test_rate=0.1):
    data = load_csv_data(iris_data)
    split_index = int(len(data)*test_rate)
    random.shuffle(data)

    return data[:-split_index],data[-split_index:]

if __name__ == '__main__':
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    train_data,test_data = load_iris_data()

    # train_random_trees(train_data,50,0.5)
    # test_random_trees(train_data)

    train(train_data)
    test(test_data)

