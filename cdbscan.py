'''
Created on 29 mars 2019

@author: galvesda
'''
import itertools
import math
import random

from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import euclidean_distances

from kdtree import KDTree
import numpy as np
from visualization import VIS


# from scipy.spatial.kdtree import KDTree
class CDBSCAN:
    
    def __init__(self, minPts, eps):
        self.minPts = minPts
        self.eps = eps

    def fit(self, X, ML, CL):
        
        X_copy = [tuple(x) for x in X]
        
        unprocessed = [True]*len(X)
        
        self.builder = KDTree(X_copy, leaf_size=self.minPts)
        self.builder.build_tree_cdbscan()
        leafnodes_tuples = self.builder.leaf_nodes
        leafnodes_list = [point_index for point_index in leafnodes_tuples]
        
#         tree = KDTree(X_copy, leafsize=self.minPts)
#         leafnodes_list = self.extract_leafnodes(tree.tree)

#         leafnodes_list = [[8,7,6],[9,10,11],[3,4,5],[0,1,2],[12,13,14],[20,24],[19,18],[21,23,22],[17,15,16]]
                
#         v = VIS()
#         v.plot_pairs(X, ML, '-', 'gray')
#         v.plot_pairs(X, CL, '-', 'tomato')
#         v.plot(X, self.get_labels(len(X), leafnodes_list))
#         v.plot_div(self.builder.divisions, '--', 'gray')
#         v.show()      
        
#         print("step 1, leaf nodes:",leafnodes_list)
#################         
#         STEP 2
        local_clusters = []
        noise_points = []
        
        for leafnode in leafnodes_list:
            unprocessed = leafnode.copy()
            not_noise_points = set()
            possible_noise_points = set()
            while len(unprocessed) > 0:
                
                pivot = random.sample(unprocessed, 1)[0]
                reachable_points = [point for point in leafnode if euclidean(X[pivot], X[point]) <= self.eps]
                
                if len(reachable_points) < self.minPts:
                    possible_noise_points.add(pivot) # NOISE
                    unprocessed.remove(pivot)
                
                elif self.check_constraints_involved2(CL, reachable_points):
                    for point in leafnode:
                        if point in unprocessed:
                            unprocessed.remove(point)
                        local_clusters.append([point])
                        not_noise_points.add(point)
                        if point in possible_noise_points:
                            possible_noise_points.remove(point)
                    break
                        
                else:
                    new_local_cluster = []
                    for point in reachable_points:
                        new_local_cluster.append(point)
                        if point in unprocessed:
                            unprocessed.remove(point)
                        not_noise_points.add(point)
                        if point in possible_noise_points:
                            possible_noise_points.remove(point)
                    local_clusters.append(new_local_cluster)
            
            noise_points += list(not_noise_points)
        
#         print("step 2, local clusters:",local_clusters)
        
#         v = VIS()
#         v.plot_pairs(X, ML, '-', 'gray')
#         v.plot_pairs(X, CL, '-', 'tomato')
#         v.plot(X, self.get_labels(len(X), local_clusters))
#         v.show()

#################                 
#         STEP 3a
        alpha_clusters = []
        for u,v in ML:
            
            target_cluster = [cluster for cluster in local_clusters if u in cluster]
            other_cluster = [cluster for cluster in local_clusters if v in cluster]
            
            if (target_cluster == [] or other_cluster == []) or target_cluster == other_cluster :
                continue # not found in any cluster
            
            target_cluster = target_cluster[0]
            other_cluster = other_cluster[0]

#             if self.check_constraints_involved(CL, target_cluster, other_cluster): # not in paper
#                 continue 
            
            if target_cluster in local_clusters:
                local_clusters.remove(target_cluster)
            
            if other_cluster in local_clusters:
                local_clusters.remove(other_cluster)
                
            target_cluster += other_cluster
            
            alpha_clusters.append(target_cluster)
        
#         v = VIS()
#         v.plot_pairs(X, ML, '-', 'gray')
#         v.plot_pairs(X, CL, '-', 'tomato')
#         v.plot(X, self.get_labels(len(X), local_clusters + alpha_clusters))
#         v.show()
        
#         print("step 3a, local_clusters + alpha_clusters:", local_clusters + alpha_clusters)
#################                 
#         STEP 3b
        
        previous_n_clusters = math.inf
         
        while len(local_clusters) < previous_n_clusters and len(local_clusters) > 0:
            
            previous_n_clusters = len(local_clusters)
            
            for local_cluster in local_clusters:
            
                closest_alpha_cluster = self.find_closest_cluster(X, CL, local_cluster, alpha_clusters)
                
                if closest_alpha_cluster is None: # is it reachable ?
                    continue
                
                closest_alpha_cluster += local_cluster
                local_clusters.remove(local_cluster)
#             previous_n_clusters = len(local_clusters) + len(alpha_clusters)
                break
                                
#         while len(local_clusters) + len(alpha_clusters) < previous_n_clusters:
#             
#             previous_n_clusters = len(local_clusters) + len(alpha_clusters)
#                     
#             for local_cluster in local_clusters:
#                 
#                 closest_alpha_cluster = self.find_closest_cluster(X, CL, local_cluster, alpha_clusters)
#                 
#                 if closest_alpha_cluster is None: # is it reachable ?
#                     continue
#                 
#                 closest_alpha_cluster += local_cluster
#                 local_clusters.remove(local_cluster)
            
        clusters = alpha_clusters + local_clusters
        
#         print("step 3b, clusters:",clusters)
        return self.get_labels(len(X), clusters)


    
    def get_labels(self, n, clusters):  
        partition = np.array([0]*n)
        cluster_index = 1
        for cluster in clusters:
            partition[cluster] = cluster_index
            cluster_index +=1  
        return partition         


    
    def density_reachable(self, X, point, leafnode):
        
        reachable = []
        for point2 in leafnode:
            dist = euclidean(X[point], X[point2])
            if dist <= self.eps:
                reachable.append(point2)
        return reachable


    
    def check_constraints_involved(self, CL, group1, group2 = None):
        
        instances = set(group1)
        if group2 is not None:
            instances.update(group2)
        
        pairs = itertools.combinations(instances,2)
        
        for u,v in pairs:
            if (u,v) in CL or (v,u) in CL:
                return True
        
        return False
    
    
    
    def check_constraints_involved2(self, CL, points):
        
        for u,v in CL:
            if u in points and v in points:
                return True
        
        return False
    
    
    
    def find_closest_cluster(self,X,CL,local_cluster,alpha_clusters):
         
        alpha_clusters_copy = alpha_clusters.copy()

        for alpha_cluster in alpha_clusters:
            if self.check_constraints_involved(CL,local_cluster,alpha_cluster):
                alpha_clusters_copy.remove(alpha_cluster)     
        
        distances = [euclidean_distances(X[local_cluster], X[cluster1]) for cluster1 in alpha_clusters_copy]
        
        min_val = math.inf
        min_index_cluster = None
        
        index = 0
        
        for distance in distances:
            val = np.min(distance,axis=None)
            
            if val < min_val and val <= self.eps and len(alpha_clusters_copy[index]) >= self.minPts: # best and reachable
                min_val = val
                min_index_cluster = index
            index += 1
        
        if min_index_cluster is None:
            return None
        
        for alpha_cluster in alpha_clusters:
            if alpha_cluster is alpha_clusters_copy[min_index_cluster]:
                return alpha_cluster
            
                
    def extract_leafnodes(self, node):
        if isinstance(node, KDTree.leafnode):
            return [node.idx.tolist()]
        elif isinstance(node, KDTree.innernode):
            return self.extract_leafnodes(node.greater) + self.extract_leafnodes(node.less)