""""""  		  	   		     		  		  		    	 		 		   		 		  
"""  		  	   		     		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		     		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		     		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		     		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		     		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		     		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		     		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		     		  		  		    	 		 		   		 		  
or edited.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		     		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		     		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		     		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		     		  		  		    	 		 		   		 		  
"""  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
import math  		  	   		     		  		  		    	 		 		   		 		  
import sys  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
import numpy as np  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
  		  	   		     		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		     		  		  		    	 		 		   		 		  
    if len(sys.argv) != 2:  		  	   		     		  		  		    	 		 		   		 		  
        print("Usage: python testlearner.py <filename>")  		  	   		     		  		  		    	 		 		   		 		  
        sys.exit(1)  		  	   		     		  		  		    	 		 		   		 		  
    # inf = open(sys.argv[1])
    # data = np.array(
    #     [list(map(float, s.strip().split(","))) for s in inf.readlines()]
    # )  #initial code

    data = np.genfromtxt(sys.argv[1], delimiter=",")
    # Skip the date column and header row if we're working on Istanbul data
    if sys.argv[1] == "Data/Istanbul.csv":
        data = data[1:, 1:]

        # compute how much of the data is training and testing
    train_rows = int(0.6 * data.shape[0])  		  	   		     		  		  		    	 		 		   		 		  
    test_rows = data.shape[0] - train_rows  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    # separate out training and testing data  		  	   		     		  		  		    	 		 		   		 		  
    train_x = data[:train_rows, 0:-1]  		  	   		     		  		  		    	 		 		   		 		  
    train_y = data[:train_rows, -1]  		  	   		     		  		  		    	 		 		   		 		  
    test_x = data[train_rows:, 0:-1]  		  	   		     		  		  		    	 		 		   		 		  
    test_y = data[train_rows:, -1]  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    print(f"{test_x.shape}")  		  	   		     		  		  		    	 		 		   		 		  
    print(f"{test_y.shape}")  		  	   		     		  		  		    	 		 		   		 		  

  	##########Linear Regression#######################
    # create a learner and train it  		  	   		     		  		  		    	 		 		   		 		  
    learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner  		  	   		     		  		  		    	 		 		   		 		  
    learner.add_evidence(train_x, train_y)  # train it  		  	   		     		  		  		    	 		 		   		 		  
    print(learner.author())  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    # evaluate in sample  		  	   		     		  		  		    	 		 		   		 		  
    pred_y = learner.query(train_x)  # get the predictions  		  	   		     		  		  		    	 		 		   		 		  
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])  		  	   		     		  		  		    	 		 		   		 		  
    print()  		  	   		     		  		  		    	 		 		   		 		  
    print("In sample results (Linear)")
    print(f"RMSE: {rmse}")  		  	   		     		  		  		    	 		 		   		 		  
    c = np.corrcoef(pred_y, y=train_y)  		  	   		     		  		  		    	 		 		   		 		  
    print(f"corr (Linear): {c[0,1]}")
  		  	   		     		  		  		    	 		 		   		 		  
    # evaluate out of sample  		  	   		     		  		  		    	 		 		   		 		  
    pred_y = learner.query(test_x)  # get the predictions  		  	   		     		  		  		    	 		 		   		 		  
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])  		  	   		     		  		  		    	 		 		   		 		  
    print()  		  	   		     		  		  		    	 		 		   		 		  
    print("Out of sample results (Linear)")
    print(f"RMSE: {rmse}")  		  	   		     		  		  		    	 		 		   		 		  
    c = np.corrcoef(pred_y, y=test_y)  		  	   		     		  		  		    	 		 		   		 		  
    print(f"corr (Linear): {c[0,1]}")
    ##########end of linear regressoin###############

    ##########DTLearner#######################
    # create a learner and train it (leaf_size=1)
    DTlearner = dt.DTLearner(leaf_size=1,verbose=True)
    DTlearner.add_evidence(train_x, train_y)  # train it
        # evaluate in sample
    pred_y = DTlearner.query(train_x)  # get the predictions
    print()
    print("In sample results (DTLearner, leaf_size=1)")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr (DTLearner, leaf_size=1): {c[0, 1]}")
        # evaluate out of sample
    pred_y = DTlearner.query(test_x)  # get the predictions
    print()
    print("Out of sample results (DTLearner, leaf_size=1)")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr (DTLearner, leaf_size=1): {c[0, 1]}")

    # create a learner and train it (leaf_size=50)
    DTlearner = dt.DTLearner(leaf_size=50, verbose=True)
    DTlearner.add_evidence(train_x, train_y)  # train it
    # evaluate in sample
    pred_y = DTlearner.query(train_x)  # get the predictions
    print()
    print("In sample results (DTLearner, leaf_size=50)")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr (DTLearner, leaf_size=50): {c[0, 1]}")

    ##########end of DTLearner###############

    ##########RTLearner#######################
    # create a learner and train it (leaf_size=1)
    RTlearner = rt.RTLearner(leaf_size=1, verbose=True)
    RTlearner.add_evidence(train_x, train_y)  # train it
    # evaluate in sample
    pred_y = RTlearner.query(train_x)  # get the predictions
    print()
    print("In sample results (RTLearner, leaf_size=1)")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr (RTLearner, leaf_size=1): {c[0, 1]}")
    # evaluate out of sample
    pred_y = RTlearner.query(test_x)  # get the predictions
    print()
    print("Out of sample results (RTLearner, leaf_size=1)")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr (RTLearner, leaf_size=1): {c[0, 1]}")

    # create a learner and train it (leaf_size=50)
    RTlearner = rt.RTLearner(leaf_size=50, verbose=True)
    RTlearner.add_evidence(train_x, train_y)  # train it
    # evaluate in sample
    pred_y = RTlearner.query(train_x)  # get the predictions
    print()
    print("In sample results (RTLearner, leaf_size=50)")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr (RTLearner, leaf_size=50): {c[0, 1]}")
    ##########end of RTLearner###############

    ##########BagLearner-lrl#######################
    # create a learner and train it (bag_size=1)
    Baglearner = bl.BagLearner(learner = lrl.LinRegLearner, kwargs = {}, bags = 1, boost = False, verbose = False)
    Baglearner.add_evidence(train_x, train_y)  # train it
    # evaluate in sample
    pred_y = Baglearner.query(train_x)  # get the predictions
    print()
    print("In sample results (BagLearner-lrl, Bag_size=1)")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr (BagLearner-lrl, Bag_size=1): {c[0, 1]}")
    # evaluate out of sample
    pred_y = Baglearner.query(test_x)  # get the predictions
    print()
    print("Out of sample results (BagLearner-lrl, Bag_size=1)")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr (BagLearner-lrl, Bag_size=1): {c[0, 1]}")

    # create a learner and train it (bag_size=20)
    Baglearner = bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=False)
    Baglearner.add_evidence(train_x, train_y)  # train it
    # evaluate in sample
    pred_y = Baglearner.query(train_x)  # get the predictions
    print()
    print("In sample results (BagLearner-lrl, Bag_size=20)")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr (BagLearner-lrl, Bag_size=20): {c[0, 1]}")
    # evaluate out of sample
    pred_y = Baglearner.query(test_x)  # get the predictions
    print()
    print("Out of sample results (BagLearner-lrl, Bag_size=20)")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr (BagLearner-lrl, Bag_size=20): {c[0, 1]}")
    ##########end of BagLearner###############

    ##########BagLearner-dt#######################
    # create a learner and train it (bag_size=1)
    Baglearner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size":1}, bags=1, boost=False, verbose=False)
    Baglearner.add_evidence(train_x, train_y)  # train it
    # evaluate in sample
    pred_y = Baglearner.query(train_x)  # get the predictions
    print()
    print("In sample results (BagLearner-dt, Bag_size=1)")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr (BagLearner-dt, Bag_size=1): {c[0, 1]}")
    # evaluate out of sample
    pred_y = Baglearner.query(test_x)  # get the predictions
    print()
    print("Out of sample results (BagLearner-dt, Bag_size=1)")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr (BagLearner-dt, Bag_size=1): {c[0, 1]}")

    # create a learner and train it (bag_size=20)
    Baglearner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size":1}, bags=20, boost=False, verbose=False)
    Baglearner.add_evidence(train_x, train_y)  # train it
    # evaluate in sample
    pred_y = Baglearner.query(train_x)  # get the predictions
    print()
    print("In sample results (BagLearner-dt, Bag_size=20)")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr (BagLearner-dt, Bag_size=20): {c[0, 1]}")
    # evaluate out of sample
    pred_y = Baglearner.query(test_x)  # get the predictions
    print()
    print("Out of sample results (BagLearner-dt, Bag_size=20)")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr (BagLearner-dt, Bag_size=20): {c[0, 1]}")
    ##########end of BagLearner-dt###############

    ##########BagLearner-rt#######################
    # create a learner and train it (bag_size=1)
    Baglearner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 1}, bags=1, boost=False, verbose=False)
    Baglearner.add_evidence(train_x, train_y)  # train it
    # evaluate in sample
    pred_y = Baglearner.query(train_x)  # get the predictions
    print()
    print("In sample results (BagLearner-rt, Bag_size=1)")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr (BagLearner-rt, Bag_size=1): {c[0, 1]}")
    # evaluate out of sample
    pred_y = Baglearner.query(test_x)  # get the predictions
    print()
    print("Out of sample results (BagLearner-rt, Bag_size=1)")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr (BagLearner-rt, Bag_size=1): {c[0, 1]}")

    # create a learner and train it (bag_size=20)
    Baglearner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 1}, bags=20, boost=False, verbose=False)
    Baglearner.add_evidence(train_x, train_y)  # train it
    # evaluate in sample
    pred_y = Baglearner.query(train_x)  # get the predictions
    print()
    print("In sample results (BagLearner-rt, Bag_size=20)")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr (BagLearner-rt, Bag_size=20): {c[0, 1]}")
    # evaluate out of sample
    pred_y = Baglearner.query(test_x)  # get the predictions
    print()
    print("Out of sample results (BagLearner-rt, Bag_size=20)")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr (BagLearner-rt, Bag_size=20): {c[0, 1]}")
    ##########end of BagLearner-rt###############
