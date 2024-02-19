
class Node:
    #傳入這兩個值,紀錄這個點負責監測的是哪個資料列和threshold是多少
    def __init__(self,index,threshold):
        #左右節點
        self.left = None
        self.right = None
        #左右節點代表的類型
        self.label_l = None
        self.label_r = None

        self.index = index
        self.threshold = threshold

    def set_left(self,tree_node):
        self.left = tree_node

    def set_right(self,tree_node):
        self.right = tree_node

    def set_label(self,groups,classes):
        #判斷是否已經沒有子節點存在
        if(self.left == None and self.right == None):
            self.label_l = self.get_max_type(groups[0],classes)
            self.label_r = self.get_max_type(groups[1],classes)

    def get_max_type(self,group,classes):
        count_map = {k:0 for k in classes}

        for row in group:
            #對應位置的值 +1
            count_map[row[-1]] += 1

        return max(count_map,key=lambda k:count_map[k])