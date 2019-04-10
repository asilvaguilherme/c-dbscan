import random

from matplotlib import colors as mcolors
import matplotlib.pyplot as plt


colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name) for name, color in colors.items())
sorted_names = [name for hsv, name in by_hsv]
# sorted_names = ["g", "y", "b", "gold", "m", "k", "gray", "salmon", "purple"]
# sorted_names.remove("w")
sorted_names = ['azure', 'beige', 'bisque', 'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 
                'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 
                'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 
                'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue', 
                'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 
                'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 
                'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 
                'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lime', 'limegreen', 'linen', 
                'magenta', 'maroon', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 
                'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 
                'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'rosybrown', 'royalblue', 
                'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'snow', 'springgreen', 'steelblue', 
                'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'yellow']

random.shuffle(sorted_names)

class VIS:
    def __init__(self):
        _, self.ax = plt.subplots(1,1, sharey=True)
    
    def plot(self, X, current_partition):
        
        for index in range(len(X)) :
            
#             if (current_partition[index] < len(sorted_names)):
            self.ax.scatter(X[index, 0], X[index, 1], c=sorted_names[current_partition[index]], marker="$"+str(index)+"$", zorder=2)
#             self.ax.text(X[index, 0], X[index, 1], str(index), color=sorted_names[current_partition[index]])
                
    
    def plot_pairs(self, X, pairs, linestyle, color):
      
        for u,v in pairs:
            self.ax.arrow(X[u, 0],X[u, 1], X[v, 0]-X[u, 0], X[v, 1]-X[u, 1], width=0.0001, color=color, linestyle=linestyle, head_length=0.0, head_width=0.0, zorder=1)
    
    def plot_div(self, pairs, linestyle, color):
      
        for u,v in pairs:
            
            u0,u1 = u
            v0,v1 = v
            
            self.ax.arrow(u0, u1, v0-u0, v1-u1, width=0.0001, color=color, linestyle=linestyle, head_length=0.0, head_width=0.0, zorder=1)
    
    
    def plot_divisions(self, divisions):
        
        _, max_x = self.ax.get_xbound()
        _, max_y = self.ax.get_ybound()
        
        for axis, value, min_val, max_val in divisions:
            if axis == 0:
                self.ax.axvline(x=value, ymin=min_val/max_y, ymax=max_val/max_y)
            elif axis == 1:
                self.ax.axhline(y=value, xmin=min_val/max_x, xmax=max_val/max_x, color='#2ca02c')
                
    def show(self):
        plt.show()