'''
Created on 25 mars 2019

@author: galvesda
'''


import math
import numpy as np
import visualization as vis

from scipy.spatial.distance import euclidean
from scipy.spatial.kdtree import KDTree

import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances


def main():
    data = pd.read_csv("exemplo-paper.csv", sep=',',header=None)
    
    ML = [(4,13),(7,10),(19,20),(16,22)]
    CL = [(5,8),(18,21)]
    
    clusterer = CDBSCAN(minPts=2, eps=3)
    partition = clusterer.fit(data.values,ML,CL)
    
#     vis.plot_pairs(data.values, ml, '-', 'gray')
#     vis.plot_pairs(data.values, cl, '-', 'tomato')
    vis.plot(data.values, partition, partition)
    vis.show()
    

class CDBSCAN:
    
    def __init__(self, minPts, eps):
        self.minPts = minPts
        self.eps = eps

    def fit(self, X, ML, CL):
        
        X_copy = X.copy()
#         status = [None]*len(X)
        unlabeled = [True]*len(X)
        
        points_cl_set = set([u for u,_ in CL] + [v for _,v in CL])
        
        tree = KDTree(X_copy, leafsize=self.minPts+1)
        leafnodes_list = self.extract_leafnodes(tree.tree)
        
#         STEP 2
        local_clusters = []
        
        for leafnode in leafnodes_list:
            for point in leafnode:
                if unlabeled[point]: # unlabeled
                    
                    reachable_points = self.density_reachable(X, point, leafnode)
                    
                    if len(reachable_points) < self.minPts:
                        unlabeled[point] = False # NOISE
                    
                    elif self.check_constraints_involved(point, points_cl_set, reachable_points):
                        new_local_cluster = []
                        new_local_cluster.append(point)
                        unlabeled[point] = False
                        for point2 in reachable_points:
                            new_local_cluster.append(point2)
                            unlabeled[point2] = False
                        local_clusters.append(new_local_cluster)
                            
                    else:
                        new_local_cluster = []
                        new_local_cluster.append(point)
                        unlabeled[point] = False
                        for point2 in reachable_points:
                            new_local_cluster.append(point2)
                            unlabeled[point2] = False
                        local_clusters.append(new_local_cluster)
        
        alpha_clusters = []
#         STEP 3a
        for u,v in ML:
            
            target_cluster = [cluster for cluster in local_clusters if u in cluster]
            other_cluster = [cluster for cluster in local_clusters if v in cluster]
            
            if target_cluster == [] or other_cluster == []:
                continue
            
            target_cluster = target_cluster[0]
            other_cluster = other_cluster[0]
            
            local_clusters.remove(target_cluster)
            
            target_cluster += other_cluster
            
            alpha_clusters.append(target_cluster)
        
#         STEP 3b
        previous_n_clusters = math.inf
         
        while len(local_clusters) + len(alpha_clusters) < previous_n_clusters:
            print(len(local_clusters) + len(alpha_clusters),previous_n_clusters) 
            previous_n_clusters = len(local_clusters) + len(alpha_clusters)
             
            for alpha_cluster in alpha_clusters:
                
                local_cluster_index = self.find_closest_cluster(X, alpha_cluster, local_clusters)
                
                if local_cluster_index is None: # is it reachable ?
                    continue
                
                local_cluster = local_clusters[local_cluster_index]
                 
                flag = True
                for point in local_cluster:
                    if not self.check_constraints_involved(point, points_cl_set, alpha_cluster):
                        flag = False
                 
                if flag:
                    alpha_cluster += local_cluster
                    local_clusters.remove(local_cluster)
            
        clusters = alpha_clusters + local_clusters
        
        return self.get_labels(len(X), clusters)
    
    def get_labels(self, n, clusters):  
        partition = np.array([-1]*n)
        cluster_index = 0
        for cluster in clusters:
            partition[cluster] = cluster_index
            cluster_index +=1           
    
    def extract_leafnodes(self, node):
        if isinstance(node, KDTree.leafnode):
            return [node.idx.tolist()]
        elif isinstance(node, KDTree.innernode):
            return self.extract_leafnodes(node.greater) + self.extract_leafnodes(node.less)

    
    def density_reachable(self, X, point, points):
        
        reachable = []
        for point2 in points:
            if point != point2:
                dist = euclidean(X[point], X[point2])
                if dist <= self.eps:
                    reachable.append(point2)
        return reachable
        
    def check_constraints_involved(self, point, points_cl_set, reachable):
        
        if point in points_cl_set:
            return True
        
        for reachable_point in reachable:
            if reachable_point in points_cl_set:
                return True
        
        return False
    
    def find_closest_cluster(self,X,cluster,clusters):
        
        distances = [euclidean_distances(X[cluster], X[cluster1]) for cluster1 in clusters]
        
        min_val = math.inf
        min_index_cluster = None
        index = 0
        
        for distance in distances:
            val = np.min(distance,axis=None)
            if val < min_val:
                min_val = val
                min_index_cluster = index
            index += 1 
        
        if min_val <= self.eps:
            min_index_cluster
        return None
        
    
if __name__== "__main__":
    main()