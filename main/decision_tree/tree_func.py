import copy
from operator import itemgetter
from typing import List

#對每欄資料各取一個 threshold 值,在後續會拿來判斷哪個 threshold 最合適
def get_threshold(index,dataset):
    threshold_list = []
    #在第一筆前加上一個特別小的資料
    threshold_list.append(dataset[0][index] - 1)

    for i in range(len(dataset)):
        if i == 0 :
            continue

        previous = dataset[i-1][index]
        current = dataset[i][index]

        #排除相同
        if(previous != current):
            threshold_list.append((previous+current)/2)

    #在最後一筆加入比最大值大的
    threshold_list.append(dataset[-1][index] + 1)

    return threshold_list

#小於門檻為左,大於為右
def data_split(index,threshold,dataset):
    left,right = [],[]

    for row in dataset:
        if row[index] < threshold:
            left.append(row)
        else :
            right.append(row)

    return left,right

#計算資料的不純度
def gini_index(groups:List[list],classes:List):
    gini = 0.
    key_index = {k:i for i,k in enumerate(classes)}

    for group in groups:
        count_list = [0 for _ in classes]

        if len(group) == 0:
            continue

        #統計每個類別的數量
        for row in group:
            index = key_index[row[-1]]
            count_list[index] += 1

        #運算不純度
        tmp = 1
        for count in count_list:
            if not count:
                continue
            else :
                tmp -= ((count/len(group)) **2)

        #考慮資料量比較長的項目權重較大,所以乘上資料長度
        tmp *= len(group)
        gini += tmp

    return gini

#確認資料內是否都相同
def data_same(dataset):
    dataset = copy.deepcopy(dataset)
    dataset = [row[:-1] for row in dataset] #把label項拿掉

    all_same = True

    for row in dataset:
        if row != dataset[0]:
            all_same = False

    return all_same

#定義樹的Node
if __name__ == '__main__':
    data = {'a': 3, 'b': 4, 'c': 5}
    print("值最大的 key:", data['a'])
    pass