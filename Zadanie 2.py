import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Zadanie 2.
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

labels[labels=='setosa'] = 0
labels[labels=='versicolor'] = 1
labels[labels=='virginica']= 2

features=features.astype(float)
# Zadanie 2.
colors = ['red','blue','yellow']
fig, ax = plt.subplots(4,4,figsize=(7,7))

ax[0,0].scatter(features[:,0],features[:,0],c=labels)
ax[0,0].set_xlabel("sepal_length")
ax[0,0].set_ylabel("sepal_length")

ax[0,1].scatter(features[:,0],features[:,1],c=labels)
ax[0,1].set_xlabel("sepal_length")
ax[0,1].set_ylabel("sepal_width")

ax[0,2].scatter(features[:,0],features[:,2],c=labels)
ax[0,2].set_xlabel("sepal_length")
ax[0,2].set_ylabel("petal_length")

ax[0,3].scatter(features[:,0],features[:,3],c=labels)
ax[0,3].set_xlabel("sepal_length")
ax[0,3].set_ylabel("petal_width")

ax[1,0].scatter(features[:,1],features[:,0],c=labels)
ax[1,0].set_xlabel("sepal_length")
ax[1,0].set_ylabel("sepal_length")

ax[1,1].scatter(features[:,1],features[:,1],c=labels)
ax[1,0].set_xlabel("sepal_length")
ax[1,0].set_ylabel("sepal_width")

ax[1,2].scatter(features[:,1],features[:,2],c=labels)
ax[1,2].set_xlabel("sepal_length")
ax[1,2].set_ylabel("petal_length")

ax[1,3].scatter(features[:,1],features[:,3],c=labels)
ax[1,3].set_xlabel("sepal_length")
ax[1,3].set_ylabel("petal_width")

ax[2,0].scatter(features[:,2],features[:,0],c=labels)
ax[2,0].set_xlabel("petal_length")
ax[2,0].set_ylabel("sepal_length")

ax[2,1].scatter(features[:,2],features[:,1],c=labels)
ax[2,1].set_xlabel("petal_length")
ax[2,1].set_ylabel("sepal_width")

ax[2,2].scatter(features[:,2],features[:,2],c=labels)
ax[2,2].set_xlabel("petal_length")
ax[2,2].set_ylabel("sepal_width")

ax[2,3].scatter(features[:,2],features[:,3],c=labels)
ax[2,3].set_xlabel("petal_length")
ax[2,3].set_ylabel("petal_width")

ax[3,0].scatter(features[:,3],features[:,0],c=labels)
ax[3,0].set_xlabel("petal_width")
ax[3,0].set_ylabel("sepal_length")

ax[3,1].scatter(features[:,3],features[:,1],c=labels)
ax[3,1].set_xlabel("petal_width")
ax[3,1].set_ylabel("sepal_width")

ax[3,2].scatter(features[:,3],features[:,2],c=labels)
ax[3,2].set_xlabel("sepal_length")
ax[3,2].set_ylabel("sepal_width")

ax[3,3].scatter(features[:,3],features[:,3],c=labels)
ax[3,3].set_xlabel("petal_width")
ax[3,3].set_ylabel("petal_width")

plt.tight_layout()

plt.savefig("Photo/zad2.png")
plt.show()