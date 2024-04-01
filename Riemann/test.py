import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
import seaborn as sns
sns.set_theme(style = 'whitegrid')
from PIL import Image
import os

L = 10
h = 0.1

NX = int(2 * L / h + 1)
# print(NX)
x_i = np.linspace(-L, L, int(NX))
# print(x_i)

# print(NX // 2)
w_i = np.zeros((3, NX))

for i in range(NX // 2):
    w_i[0, i] = 13
    w_i[2, i] = 10 / (5 / 3 - 1)

for i in range((NX // 2), NX):
    w_i[0, i] = 1.3
    w_i[2, i] = 1 / (5 / 3 - 1)

# print(w_i)
v_i = np.zeros((3, NX))
v_i[0, :] = w_i[0, :] 
v_i[1, :] = w_i[1, :]
v_i[2, :] = w_i[2, :] / w_i[0, :]
print(v_i[2, :].max())

t = np.array([[-1, -1], [2, -1]])
print(w_i)
print(w_i[:, 0])

print(np.isclose(1, 1)[0])