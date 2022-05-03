import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator




def check_groups_strategy(groups_strategy, num):
    #! 检查分组策略是否基本正确
    s = []
    for g in groups_strategy:
        s += g
    if len(s) == num:
        return True
    else:
        False





###############* plot
def plot_series(x, xticks=None, path=None):
    plt.figure()
    ax=plt.gca()
    plt.plot(x)
    if xticks:
        ax.xaxis.set_major_locator(MultipleLocator(10))
        plt.tick_params(labelsize=5)
        plt.xticks(rotation=90)
    if path:
        plt.savefig(path)
    plt.show()



def plot_series_array(x, path=None):
    if len(x.shape) != 2:
        print("Input data is not array.")
        return 
    plt.figure()
    for i in range(x.shape[0]):
        plt.subplot(x.shape[0], 1, i+1)
        plt.plot(x[i])
    if path:
        pass
    plt.show()