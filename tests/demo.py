import os, sys
os.chdir(sys.path[0])
sys.path.append("..")

import matplotlib.pyplot as plt
from src.ssa.ssa import *
from data.load import *
import numpy as np




path = "data/WHO-COVID-19-global-data.csv"
country = "China"
L = 85
stride = 1
tao = 1


#!
# data_dict = read_data(path, country)
# x = np.log(data_dict["New_cases"]+1)

t = np.arange(0, 200, 0.1)
# x = 5*t**11 + 9
# x = np.exp(-0.01*t)
# fs = 2
# num = 1000
# t = np.arange(0, num)
# t = t/fs
# omega = 1/25
# x = np.sin(2*np.pi*omega*t)


L = 85
t = np.arange(0, 340, 1)
x = np.exp(t/400) + np.sin(2*np.pi*t/17) + 0.5*np.sin(2*np.pi*t/10) + np.random.randn(340)
#!

#!
model = SSA(L, stride, tao)
model.embedding(x)
model.decomposition_safe()
model.reconstruction()
model.grouping(option="SC", params={"n_clusters": 3})



plot_series(x, np.sum(model.components_series[0:1], axis=0))
plot_series_array(x, model.components_series_grouped)

#! 绘制w-相关矩阵
w_matrix = w_correlation_matrix(model.weights, model.components_series)
plot_wmatrix(w_matrix)

# 绘制两个特征向量
# txt = [str(i) for i in range(len(model.eigenvector_left[:, 0]))]
# x = model.eigenvector_left[:, 0]
# y = model.eigenvector_left[:, 1]
# plt.figure()
# plt.scatter(x, y)
# for i in range(10):
#     plt.annotate(txt[i], xy = (x[i], y[i]), xytext = (x[i]+0.01, y[i]+0.01))
# plt.show()

# 估计频率
# angle = polar_angle(model.eigenvector_left[:, 0:1], model.eigenvector_left[:, 1:2])
# omega_estimation = freq_estimation(model.eigenvector_left[:, 0:1], model.eigenvector_left[:, 1:2])
# print(omega_estimation*fs, omega)


trend_index, trend = model.extract_trend(omega=1/24, thres=0.9)
plt.figure()
plt.plot(x)
plt.plot(trend)
plt.show()


harmonic_index, harmonics = model.extract_harmonics()
print(harmonic_index)
plt.figure()
plt.plot(x)
plt.plot(harmonics[0])
plt.show()


xx = recurrent_forcast(model.components_series_grouped[1], model.eigenvector_left[:, model.groups_strategy[1]], 100)
plt.figure()
plt.plot(np.concatenate([model.components_series_grouped[1], xx]))
plt.plot([i for i in range(model.length, model.length+len(xx))], xx, linewidth=3)
plt.show()

y = vector_predict(model.components_array_grouped[1], model.eigenvector_left[:, model.groups_strategy[1]], 20)
plt.figure()
plt.plot(np.concatenate([model.components_series_grouped[1], y]))
plt.plot([i for i in range(model.length, model.length+len(y))], y, linewidth=3)
plt.show()

