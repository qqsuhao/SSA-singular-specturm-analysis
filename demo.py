from model.ssa import *
from data.load import *
import numpy as np



path = "data/WHO-COVID-19-global-data.csv"
country = "China"
L = 120
stride = 1
tao = 1


#!
data_dict = read_data(path, country)
x = np.log(data_dict["New_cases"]+1)

#!
# plot_series(x)

#!
model = SSA(L, stride, tao)
model.embedding(x)
model.decomposition()
model.grouping()
model.reconstruction()


# plot_series_array(model.components_series)
# plot_series(np.sum(model.components_series[0:6], axis=0))

plt.figure()
plt.plot(x)
plt.plot(np.sum(model.components_series[0:1], axis=0))
plt.legend(["origin series", "trend"])
plt.show()


plt.figure()
plt.plot(x)
plt.plot(np.sum(model.components_series[0:10], axis=0))
plt.legend(["origin series", "trend"])
plt.show()