'''
Created on 25 mars 2019

@author: galvesda
'''


import random

from active_semi_clustering.active.pairwise_constraints.example_oracle import ExampleOracle, MaximumQueriesExceeded
from scipy.special import comb
from sklearn.cluster.dbscan_ import DBSCAN

from cdbscan import CDBSCAN
import numpy as np
import pandas as pd
from visualization import VIS


def main():

    data = pd.read_csv("r_db7.data", sep=',',header=None)
#     data = pd.read_csv("MyIris.data", sep=',',header=None)
 
    num_attrib = data.values.shape[1] - 1
    X = data.values[:,0:num_attrib]
    Y = data.values[:,-1]
    ML, CL = random_generator(X, Y, 50)
    print("ML",ML)
    print("CL",CL)
    
    best_config = param_opt(X, Y, ML, CL, [5,10,20,30,40,50], [1,1.5,2,2.5,3,3.5,4,4.5,5])
    print(best_config)
#     best_config = exec_dbscan(X, Y, [5,10,20,30,40,50], [1,1.5,2,2.5,3,3.5,4,4.5,5])
#     print(best_config)
    
def one_execution():
    
    X = pd.read_csv("exemplo-paper.data", sep=',',header=None).values
    Y = np.array([1,1,1,1,1,1,2,2,2,2,2,2,1,1,1,4,4,4,3,3,3,4,4,4,3])
    ML = [(4,13),(7,10),(19,20),(17,22)]
    CL = [(5,8),(18,21)]
    clusterer = CDBSCAN(minPts=2, eps=3)
    partition = clusterer.fit(X,ML,CL)
    
    print(partition)
    print(rand_index_score(Y.astype(int), partition.astype(int)))

    v = VIS()
    v.plot_pairs(X, ML, '-', 'gray')
    v.plot_pairs(X, CL, '-', 'tomato')
    v.plot(X, partition)
    v.show()
    
def exec_dbscan(X, Y, range_minPts, range_eps, repetitions=10):
    
    best = None
    config = None
    
    for minPts in range_minPts:
        for eps in range_eps:
            
            performance_list = []
            
            for _ in range(repetitions):
                clusterer = DBSCAN(eps=eps, min_samples=minPts).fit(X)
                partition = clusterer.labels_
                
                partition = [x+1 for x in partition]
                
                performance_list.append(rand_index_score(Y.astype(int), partition))
            
            avg_performance = np.mean(performance_list)
            print(minPts, eps, avg_performance)
                
            if config is None or best < avg_performance:
                best = avg_performance
                config = (minPts, eps)
                
    return config
    
    

def param_opt(X, Y, ML, CL, range_minPts, range_eps, repetitions=10):
    
    best = None
    config = None
    
    for minPts in range_minPts:
        for eps in range_eps:
            
            performance_list = []
            
            for _ in range(repetitions):
                clusterer = CDBSCAN(minPts, eps)
                partition = clusterer.fit(X,ML,CL)
                
                performance_list.append(rand_index_score(Y.astype(int), partition.astype(int)))
            
            avg_performance = np.mean(performance_list)
            print(minPts, eps, avg_performance)
                
            if config is None or best < avg_performance:
                best = avg_performance
                config = (minPts, eps)
                
    return config

    
def random_generator(shuffled_X, shuffled_Y, n_queries):
    
    oracle = ExampleOracle(shuffled_Y, max_queries_cnt=n_queries)
    
    ml, cl = [], []
        
    while True:
        choices = random.sample(range(len(shuffled_X)), 2)
        u,v = choices[0], choices[1]
        
        try:
            if oracle.query(u,v) :
                ml.append((u,v))
            else :
                cl.append((u,v))
                
        except MaximumQueriesExceeded:
            break
    
    return ml, cl

def rand_index_score(clusters, classes):
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)
    
if __name__== "__main__":
    main()