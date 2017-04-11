import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('/Users/lgnonato/Meusdocs/Cursos/CUSP-GX-5006/Data/manhattan-dof.csv',index_col=False,delimiter=';')

####################
# Classification
####################

#df_x = df.ix[:,3:5]
#df_y = df.ix[:,1]
#
#X = df_x.as_matrix()
#Y = df_y.as_matrix()
#
## --------------------
## K-fold CV
## --------------------
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=3)
#ts = Y_test.shape[0]
#
## Train
#dtc = DecisionTreeClassifier()
#dtc.fit(X_train, Y_train)
#
## Predicting
#ypred = dtc.predict(X_test)
#
## Finding mispredicted samples
#vecerror = np.asarray([int(ypred[i] != Y_test[i]) for i in range(0,ts)])
#error = np.sum(vecerror)
#ccidx = np.where(vecerror == 0)
#mcidx = np.where(vecerror == 1)
#
#print("----------Decision Tree Classfication----------")
#print(error, "misclassified data out of", ts, "(",error/ts,"%)")
#
#plt.scatter(X_train[:,0],X_train[:,1], c=Y_train, vmin=0, vmax = 4, cmap='Accent')
#plt.scatter(X_test[mcidx,0],X_test[mcidx,1], marker='s', s=70, c=ypred[mcidx], vmin=0, vmax = 4, cmap='Accent')
#plt.scatter(X_test[mcidx,0],X_test[mcidx,1], marker='s', s=20, c=Y_test[mcidx], vmin=0, vmax = 4, cmap='Accent')


####################
# Regression
####################

df_x = df.ix[:,3:5]
df_y = df.ix[:,5]

X = df_x.as_matrix()
Y = df_y.as_matrix()

# --------------------
# K-fold CV
# --------------------
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=3)
ts = Y_test.shape[0]

# Train
dtc = DecisionTreeRegressor()
dtc.fit(X_train, Y_train)

# Predicting
ypred = dtc.predict(X_test)

# Computint prediction error
error = np.mean((ypred - Y_test) ** 2)

print("----------Decision Tree Regression----------")
print("Regression error:", error)

c=1
plt.scatter(X_train[:,c],Y_train[:])
plt.scatter(X_test[:,c],Y_test[:], color='yellow')
plt.scatter(X_test[:,c],ypred[:], color='red')
for i in range (ts):
    x = X_test[i,c]
    yt = Y_test[i]
    yp = ypred[i]
    plt.plot([x,x], [yt,yp], color="gray")


plt.show()
