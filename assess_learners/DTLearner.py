import numpy as np
class DTLearner(object):
    def __init__(self, leaf_size=1,verbose=False):
        self.leaf_size=leaf_size
        self.verbose=verbose


    def author(self):
        return "sli761"
    def find_best_feat(self,data_x,data_y):
        x_num=data_x.shape[1]
        corr_arr=[]
        for i in range(x_num):
            corr_arr.append(np.corrcoef(data_x[:,i],data_y)[0,1]) #should use corrcoef, but correlate can generate correct result
            # corr_arr.append(np.correlate(data_x[:, i], data_y))
        return np.argmax(np.absolute(corr_arr))
    def find_split_val(self,data_x,data_y):
        return np.median(data_x[:,self.find_best_feat(data_x,data_y)])

    def build_tree(self,data):
        data_x = data[:, :data.shape[1] - 1]
        data_y = data[:, -1]
        if data.shape[0] <= self.leaf_size:
            return np.array([[-1, np.mean(data[:, -1]), np.nan, np.nan]])
        if np.unique(data[:, -1]).shape[0] == 1:
            return np.array([[-1, np.unique(data[:, -1])[0], np.nan, np.nan]])
        else:
            best_feat = self.find_best_feat(data_x, data_y)
            split_val=np.median(data_x[:,best_feat])
            if np.array_equal(data[data[:, best_feat] <= split_val], data):
                return np.array([[-1, np.mean(data[:, -1]), np.nan, np.nan]])
            left_tree = self.build_tree(data[data[:, best_feat] <= split_val])
            right_tree = self.build_tree(data[data[:, best_feat] > split_val])
            root = np.array([best_feat, split_val, 1, left_tree.shape[0] + 1])
            return np.vstack([root, left_tree, right_tree])



    # for debug use:
    # a = DTLearner(leaf_size=1, verbose=True)
    # a = DTLearner(leaf_size=1, verbose=False)
    # res=a.build_tree(data)
    # res=a.add_evidence(data_x,data_y)
    # x=data_x
    # a.query(x)
    # comp=a.query_result-data_y
    # x[[comp!=0]]
    # np.array(a.query_result)[comp!=0]




    def add_evidence(self, data_x, data_y):
        data=np.column_stack((data_x,data_y))
        self.tree=self.build_tree(data)
        if self.verbose==True:
            return self.tree


    def query(self, points):
        output=[]
        for index,Xs in enumerate(points):
            start_point=0
            factor=self.tree[start_point,0]
            while factor!=-1:
                if Xs[int(factor)]<=self.tree[int(start_point),1]:
                    start_point=start_point+1
                else:
                    start_point = start_point+self.tree[int(start_point), 3]
                factor = self.tree[int(start_point), 0]

            output.append(self.tree[int(start_point),1])
        self.query_result=output
        return output
        if self.verbose==True:
            print(output)
            return self.query_result

if __name__ == "__main__":
    print("")