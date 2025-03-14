import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Zadanie 1.
data = np.loadtxt('iris.csv',delimiter=',',dtype='object')
print(data.shape)
header_val = data[0]
print(header_val)
data = data[1:]
print(data.shape)
features = data[:,0:4]
labels = data[:,-1]
#print(features)
#print(labels)
#print(lab)

labels[labels=='setosa'] = 0
labels[labels=='versicolor'] = 1
labels[labels=='virginica']= 2

class_0 = features[labels==0]
class_1 = features[labels==1]
class_2 = features[labels==2]

features=features.astype(float)

fig, ax = plt.subplots(1,1,figsize=(7,7))
ax.scatter(features[:,1],features[:,0],c=labels)


plt.xlabel("sepal_length")
plt.ylabel("sepal_width")
plt.savefig("Photo/zad1.png")
plt.show()

