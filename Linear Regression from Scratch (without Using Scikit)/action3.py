import regression
from numpy import *
import numpy as np
theta = [[np.random.rand()],[np.random.rand()]]
xArr,yArr=regression.loadDataSet('ex0.txt')
past_thetas, past_costs = regression.gradient_descent(xArr,yArr , 500, 0.01)
ws = past_thetas[-1]

xMat=mat(xArr)
yMat=mat(yArr)
yHat = xMat*ws
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
xCopy=xMat.copy()
xCopy.sort(0)
yHat=xCopy*ws
ax.plot(xCopy[:,1],yHat)
#plt.plot(past_costs)
plt.show()
yHat = xMat*ws
print(corrcoef(yHat.T, yMat))
print(ws)