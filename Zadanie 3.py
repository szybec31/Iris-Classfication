import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

#Zadanie 3.
data = np.loadtxt('iris.csv',delimiter=',',dtype='object')
print("Read data structure: {}".format(data.shape))
header_val = data[0]
print("Headers: {}".format(header_val))
data = data[1:]
print("Read data structure: {}".format(data.shape))
features = data[:,2:4]  # features
labels = data[:,-1]     # name classes [setosa,versicolor,virginica]
#print(features)
#print(labels)

# replace name classes to 0,1,2
labels[labels=='setosa'] = 0
labels[labels=='versicolor'] = 1
labels[labels=='virginica']= 2
#print(labels)

features=features.astype(float)

# New Object details
new_object = np.array([3.1,1.2]).reshape(1,2)

# split data to chart
class_0 = features[labels==0]
class_1 = features[labels==1]
class_2 = features[labels==2]

colors = []
for i in range(0,50):
    colors.append("red")
for i in range(0, 50):
    colors.append("blue")
for i in range(0, 50):
    colors.append("green")
print("Length of colors Tab: {}".format(len(colors)))

# create new chart
fig, ax = plt.subplots(1,1,figsize=(7,7))

ax.scatter(features[:,0],features[:,1],c=colors,alpha=0.15)     # add iris data to chart

m1 = [np.mean(class_0[:,0]),np.mean(class_0[:,1])]  # calculate class 0 centroid
m2 = [np.mean(class_1[:,0]),np.mean(class_1[:,1])]  # calculate class 1 centroid
m3 = [np.mean(class_2[:,0]),np.mean(class_2[:,1])]  # calculate class 2 centroid
cent = np.array([m1,m2,m3])

# add centroids to chart
ax.scatter(np.mean(class_0[:,0]),np.mean(class_0[:,1]),color="red",label="setosa class0",alpha=1)
ax.scatter(np.mean(class_1[:,0]),np.mean(class_1[:,1]),color="blue",label="versicolor class1",alpha=1)
ax.scatter(np.mean(class_2[:,0]),np.mean(class_2[:,1]),color="green",label="virginica class2",alpha=1)
ax.scatter(3.1,1.2,color='black',label="New Object",alpha=1)    # New Obcject

plt.title("New Iris Prediction")
plt.xlabel("petal length [cm]")
plt.ylabel("petal width [cm]")
plt.legend()
plt.tight_layout()

# calculate distance from each centroid
met1 = cdist(cent,new_object)
print("Distance from each centroid: ")
print(met1)
min_val = np.argmin(met1)
print("Nearest class: class {}".format(min_val))

if min_val == 0:
    print("predicted class 0")
elif min_val == 1:
    print("predicted class 1")
elif min_val == 2:
    print("predicted class 2")

plt.savefig("Photo/zad3.png")
plt.show()