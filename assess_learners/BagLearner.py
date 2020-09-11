import numpy as np
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import LinRegLearner as lrl

class BagLearner(object):
    def __init__(self, learner, kwargs = {}, bags = 20, boost = False, verbose = False):
        self.learners=[]
        self.bags=bags
        self.kwargs=kwargs
        self.boost=boost
        self.verbose=verbose
        for i in range(self.bags):
            self.learners.append(learner(**kwargs))

    def author(self):
        return "sli761"

    def add_evidence(self, data_x, data_y):
        data=np.column_stack((data_x, data_y))
        row_size = data.shape[0]
        for i in range(self.bags):
            data_temp = data[np.random.randint(row_size, size=row_size), :]
            data_temp_x = data_temp[:, :data.shape[1] - 1]
            data_temp_y = data_temp[:, -1]
            self.learners[i].add_evidence(data_temp_x, data_temp_y)
    def query(self, points):
        pred_y=[]
        for i in range(self.bags):
            temp=self.learners[i].query(points)
            pred_y.append(temp)
        res=np.mean(pred_y,axis=0)
        if self.verbose:
            print(res.shape)
        return res


if __name__ == "__main__":
    print("")