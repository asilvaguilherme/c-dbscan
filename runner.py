'''
Created on 25 mars 2019

@author: galvesda
'''


from cdbscan import CDBSCAN
import pandas as pd
import visualization as vis


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
    

    
if __name__== "__main__":
    main()