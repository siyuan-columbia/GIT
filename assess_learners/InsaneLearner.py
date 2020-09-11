import numpy as np
import BagLearner as bl
import LinRegLearner as lrl

class InsaneLearner(object):
    def __init__(self, verbose):
        self.learners = []

    def author(self):
        return 'sli761'

    def add_evidence(self, data_x, data_y):
        for i in range(20):
            learner = bl.BagLearner(learner = lrl.LinRegLearner, kwargs = {}, bags = 20, boost = False, verbose = False)
            learner.add_evidence(data_x, data_y)
            self.learners.append(learner)

    def query(self, points):
        pred_y = []
        for learner in self.learners:
            pred_y.append(learner.query(points))
        return np.mean(pred_y, axis= 0)