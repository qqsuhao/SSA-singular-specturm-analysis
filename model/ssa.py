import numpy as np



class SSA:
    def __init__(self, L, stride=1, tao=1):
        #* L：相空间的维度
        #* stride：窗口移动步长
        #* tao：延时长度
        #* 默认stride=1, tao=1，相空间为轨迹矩阵
        self.L, self.stride, self.tao = L, stride, tao
    

    def embedding(self, x):
        #*   构造相空间，特殊情况为轨     迹矩 阵
        pass


    def decomposition(self):
        pass

    
    def reconstruction(self):
        pass

