import KNN
import matplotlib
import matplotlib.pyplot as plt
from numpy import *

#group,labels = KNN.createDataset()
#result = KNN.classify0([0,0],group,labels,1)

datingDataMat,DatingLabels = KNN.file2matrix('datingTestSet2.txt')
#print(datingDataMat,DatingLabels)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(DatingLabels),15.0*array(DatingLabels))
plt.show()
