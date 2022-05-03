import numpy as np
from model.utils import *


class SSA:
    def __init__(self, L, stride=1, tao=1):
        #* L：相空间的维度
        #* stride：窗口移动步长
        #* tao：延时长度
        #* 默认stride=1, tao=1，相空间为轨迹矩阵
        self.L, self.stride, self.tao = L, stride, tao
        self.K = 0                  #! 延时向量的个数
        self.length = 0             #! 时间序列长度
        self.phase_space = np.array([]) 
        self.phase_space_index = np.array([])       #! 用于记录时间序列下标的位置，用于将矩阵转换回时间序列
        self.components_array = np.array([])
        self.components_groups = np.array([])        #! 分组结果
        self.components_series = np.array([])
        self.groups_strategy = []
    

    def embedding(self, x):
        #*   构造相空间，特殊情况为轨迹矩阵
        self.L_span = (self.L-1)*(self.tao-1) + self.L
        if len(x) < self.L_span:
            assert "The length of time series is too short to embedding."

        self.length = len(x)
        x_index = np.arange(0, len(x), 1)
        cols, rows = (len(x)-self.L_span) // self.stride + 1, self.L
        self.K = cols
        self.phase_space = np.zeros((rows, cols))
        self.phase_space_index = np.zeros_like(self.phase_space)
        for r in range(cols):
            index = [i for i in range(r*self.stride, r*self.stride+self.L_span, self.tao)]
            self.phase_space[:, r] = x[index]
            self.phase_space_index[:, r] = x_index[index]


    def decomposition(self):
        #* 对相空间矩阵进行SVD分解并获取子成份
        if len(self.phase_space) == 0:
            assert "Please take embedding first."
        S = self.phase_space @ self.phase_space.T
        Lambda, U = np.linalg.eig(S)
        V = self.phase_space.T @ U / np.sqrt(Lambda)
        self.components_array = np.zeros((len(Lambda), self.L, self.K))
        for i in range(len(Lambda)):
            self.components_array[i] = np.sqrt(Lambda[i]) * (U[0:,i:i+1] @ V[0:, i:i+1].T)



    def grouping(self):
        #* 对分解结果进行分组
        self.groups_strategy = [[i] for i in range(len(self.components_array))]
        if not check_groups_strategy(self.groups_strategy, self.L):
            assert "Grouping strategies are wrong."



    def convert_array_to_series(self, array):
        #* 将矩阵转换回时间序列
        if array.shape != self.phase_space_index.shape:
            assert "Something Wrong when embedding or decomposition."
        
        series = np.zeros((self.length, ))
        for i in range(self.length):
            index = np.where(self.phase_space_index==i)
            if len(index[0]) == 0:
                print("Warning! some values were left when embedding")
                series[i] = 0
            else:
                series[i] = np.mean(array[index])
        return series


    def reconstruction(self):
        #* 根据分组结果，经分组后的子成分还原回时间序列。
        self.components_groups = np.zeros((len(self.groups_strategy), self.L, self.K))
        self.components_series = np.zeros((len(self.groups_strategy), self.length))
        for i in range(len(self.groups_strategy)):
            group = np.sum(self.components_array[self.groups_strategy[i]], axis=0)
            self.components_groups[i] = group
            self.components_series[i] = self.convert_array_to_series(group)          #! 从矩阵转换回时间序列

