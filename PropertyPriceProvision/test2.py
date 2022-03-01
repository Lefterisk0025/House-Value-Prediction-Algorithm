import numpy as np

y = 0.3678
wT = np.ones(4)
x = np.array([0.678, 0.345, 0.765, 0.901])

features = [[0.678, 0.345, 0.765, 0.901], [0.678, 0.345, 0.765, 0.901], [0.678, 0.345, 0.765, 0.901]]
for z in range(0,4):
    row = [i[z] for i in features]
    print(row)

dot = np.dot(wT, x)

Jw = y - dot
print(Jw)