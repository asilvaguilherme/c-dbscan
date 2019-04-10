import math

import numpy as np


class KDTree:
    
    def __init__(self, data_points, leaf_size):
        self.leaf_size = leaf_size
        self.leaf_nodes = []
        self.points = data_points
        self.initial_indexes = [i for i in range(len(self.points))]
        self.dim = len(data_points[0])
        self.divisions = []

#     def build_tree(self, indexes=None, i=0):
#         
#         if indexes is None:
#             indexes = self.initial_indexes
#         
#         if math.ceil(len(indexes)/2) <= self.leaf_size:
#             node = (indexes)
#             self.leaf_nodes.append(node)
#             return node
#         
#         else:
#             tuples = [(index,self.points[index]) for index in indexes]
#             tuples.sort(key=lambda x: x[1][i])
#             
#             indexes = [index for (index,_) in tuples]
#             values = [value for (_,value) in tuples]
#             
# #             i = (i + 1) % self.dim
#             
#             half = len(indexes) >> 1
#             
#             i = self.maxspread(np.array(values))
#             
#             j = (i + 1) % self.dim
#             
#             values_arr = np.array(values)
#             
#             self.divisions.append((i,values[half + 1][i],np.min(values_arr[:,j]),np.max(values_arr[:,j])))
#             
#             return (self.build_tree(indexes[: half + 1], i),self.build_tree(indexes[half + 1:], i))
        
    def build_tree_cdbscan(self, indexes=None, i=0):
        
        if indexes is None:
            indexes = self.initial_indexes
        
        if math.ceil(len(indexes)/2) < self.leaf_size:
            node = (indexes)
            self.leaf_nodes.append(node)
            return node
        
        else:
            tuples = [(index,self.points[index]) for index in indexes]
            tuples.sort(key=lambda x: x[1][i])
            
            indexes = [index for (index,_) in tuples]
            values = [value for (_,value) in tuples]
            
            half = len(indexes) >> 1
            
            i = self.maxspread(np.array(values))
            
            values_arr = np.array(values)

            j = (i + 1) % self.dim
            if i == 0:
                u = (values[half + 1][i],np.min(values_arr[:,j]))
                v = (values[half + 1][i],np.max(values_arr[:,j]))
                self.divisions.append((u,v))
            elif i == 1:
                u = (np.min(values_arr[:,j]),values[half + 1][i])
                v = (np.max(values_arr[:,j]),values[half + 1][i])
                self.divisions.append((u,v))
                
#             print((i,values[half + 1][i],np.min(values_arr[:,j]),np.max(values_arr[:,j])))
            return (self.build_tree_cdbscan(indexes[: half + 1], i),self.build_tree_cdbscan(indexes[half + 1:], i))
        
        
    def maxspread(self, values):
        
        best_index = -1
        best_value = -math.inf
        
        for d in range(self.dim):
#             max_value = max(values[d]) 
#             min_value = min(values[d])
#             diff = max_value - min_value
            diff = np.var(values[:,d])
            
            if diff > best_value:
                best_index = d
                best_value = diff
            
        return best_index