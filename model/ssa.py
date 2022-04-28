import numpy as np



class SSA:
    def __init__(self, L, stride=1, tao=1):
        #* L：相空间的维度
        #* stride：窗口移动步长
        #* tao：延时长度
        #* 默认stride=1, tao=1，相空间为轨迹矩阵
        self.L, self.stride, self.tao = L, stride, tao
        self.phase_space = np.array([]) 
    

    def embedding(self, x):
        #*   构造相空间，特殊情况为轨迹矩阵
        self.L_span = (self.L-1)*(self.tao-1) + self.L
        if len(x) < self.L_span:
            assert "The length of time series is too short to embedding"
        rows, cols = (len(x)-self.L_span) // self.stride + 1, self.L
        self.phase_space = np.zeros((rows, cols))
        for r in range(rows):
            index = [i for i in range(r*self.stride, r*self.stride+self.L_span, self.tao)]
            self.phase_space[r, :] = x[index]


    def decomposition(self):
        pass

    
    def reconstruction(self):
        pass

