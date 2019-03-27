import random

from matplotlib import colors as mcolors

import matplotlib.pyplot as plt

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name) for name, color in colors.items())
sorted_names = [name for hsv, name in by_hsv]
random.shuffle(sorted_names)
sorted_names.remove("w")
sorted_names = ["g", "y", "b", "gold", "m", "k", "gray", "salmon", "purple"]
red_color = "r"

n_plots = 1
_, ax = plt.subplots(n_plots,n_plots, sharey=True)

def plot(X, Y, current_partition):
        
#     projection = TSNE(n_components=2,early_exaggeration=20).fit_transform(X)
# 
#         for i in range(len(X)):
#             plt.scatter(projection[i, 0], projection[i, 1], color= sorted_names[current_partition[i]])

   
    for index in range(len(X)) :
        ax.scatter(X[index, 0], X[index, 1], c=sorted_names[current_partition[index]], marker='.', zorder=2)
            

def plot_pairs(X, pairs, linestyle, color):
  
    for u,v in pairs:
        if X[v, 0]!=X[u, 0] and X[v, 1] != X[u, 1]:
            ax.arrow(X[u, 0],X[u, 1], X[v, 0]-X[u, 0], X[v, 1]-X[u, 1], width=0.0001, color=color, linestyle=linestyle, head_length=0.0, head_width=0.0, zorder=1)



def show():
    plt.show()