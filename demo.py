from model.ssa import *
from data.load import *
import numpy as np

x = np.arange(0, 100, 1)

model = SSA(10, 1, 1)
model.embedding(x)
model.decomposition()

print(model.phase_space)
print(np.sum(model.component, axis=0) - model.phase_space)