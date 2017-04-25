import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('/Users/lgnonato/Meusdocs/Cursos/CUSP-GX-5006/Data/manhattan-dof.csv',index_col=False,delimiter=';')

df_x = df.ix[:,3:6]
df_y = df.ix[:,1]

X = df_x.as_matrix()
Y = df_y.as_matrix()

# --------------------
# K-fold CV
# --------------------
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=3)
ts = Y_test.shape[0]

# --------------------
# Decision Tree Classifier
# --------------------
# Train
dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)

# Predicting
dtc_pred = dtc.predict(X_test)

# Finding mispredicted samples
dtc_verror = np.asarray([int(dtc_pred[i] != Y_test[i]) for i in range(0,ts)])
dtc_error = np.sum(dtc_verror)
dtc_ccidx = np.where(dtc_verror == 0)
dtc_mcidx = np.where(dtc_verror == 1)

print("----------Decision Tree Classfication----------")
print(dtc_error, "misclassified data out of", ts, "(",dtc_error/ts,"%)")

#plt.figure(1)
#plt.scatter(X_train[:,0],X_train[:,1], c=Y_train, vmin=0, vmax = 4, cmap='Accent')
#plt.scatter(X_test[dtc_mcidx,0],X_test[dtc_mcidx,1], marker='s', s=60, c=dtc_pred[dtc_mcidx], vmin=0, vmax = 4, cmap='Accent')
#plt.scatter(X_test[dtc_mcidx,0],X_test[dtc_mcidx,1], marker='s', s=10, c=Y_test[dtc_mcidx], vmin=0, vmax = 4, cmap='Accent')

# --------------------
# Bagging Classifier
# --------------------
bagb = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=20)
#adab = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=20,learning_rate=1.5,algorithm="SAMME")
bagb.fit(X_train,Y_train)

# Predicting
bagb_pred = bagb.predict(X_test)

# Finding mispredicted samples
bagb_verror = np.asarray([int(bagb_pred[i] != Y_test[i]) for i in range(0,ts)])
bagb_error = np.sum(bagb_verror)
bagb_ccidx = np.where(bagb_verror == 0)
bagb_mcidx = np.where(bagb_verror == 1)

print("----------Bagging Classfication----------")
print(bagb_error, "misclassified data out of", ts, "(",bagb_error/ts,"%)")

#plt.figure(2)
#plt.scatter(X_train[:,0],X_train[:,1], c=Y_train, vmin=0, vmax = 4, cmap='Accent')
#plt.scatter(X_test[bagb_mcidx,0],X_test[bagb_mcidx,1], marker='s', s=60, c=bagb_pred[bagb_mcidx], vmin=0, vmax = 4, cmap='Accent')
#plt.scatter(X_test[bagb_mcidx,0],X_test[bagb_mcidx,1], marker='s', s=10, c=Y_test[bagb_mcidx], vmin=0, vmax = 4, cmap='Accent')

# --------------------
# AdaBoost Classifier
# --------------------
adab = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=20)
#adab = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=20,learning_rate=1.5,algorithm="SAMME")
adab.fit(X_train,Y_train)

# Predicting
adab_pred = adab.predict(X_test)

# Finding mispredicted samples
adab_verror = np.asarray([int(adab_pred[i] != Y_test[i]) for i in range(0,ts)])
adab_error = np.sum(adab_verror)
adab_ccidx = np.where(adab_verror == 0)
adab_mcidx = np.where(adab_verror == 1)

print("----------AdaBoosting Classfication----------")
print(adab_error, "misclassified data out of", ts, "(",adab_error/ts,"%)")

#plt.figure(3)
#plt.scatter(X_train[:,0],X_train[:,1], c=Y_train, vmin=0, vmax = 4, cmap='Accent')
#plt.scatter(X_test[adab_mcidx,0],X_test[adab_mcidx,1], marker='s', s=60, c=adab_pred[adab_mcidx], vmin=0, vmax = 4, cmap='Accent')
#plt.scatter(X_test[adab_mcidx,0],X_test[adab_mcidx,1], marker='s', s=10, c=Y_test[adab_mcidx], vmin=0, vmax = 4, cmap='Accent')

# --------------------
# Gradient Boosting Classifier
# --------------------
gradb = GradientBoostingClassifier(max_depth=5, n_estimators=30)
gradb.fit(X_train,Y_train)

# Predicting
gradb_pred = gradb.predict(X_test)

# Finding mispredicted samples
gradb_verror = np.asarray([int(gradb_pred[i] != Y_test[i]) for i in range(0,ts)])
gradb_error = np.sum(gradb_verror)
gradb_ccidx = np.where(gradb_verror == 0)
gradb_mcidx = np.where(gradb_verror == 1)

print("----------Gradient Boosting Classfication----------")
print(gradb_error, "misclassified data out of", ts, "(",gradb_error/ts,"%)")

#plt.figure(4)
#plt.scatter(X_train[:,0],X_train[:,1], c=Y_train, vmin=0, vmax = 4, cmap='Accent')
#plt.scatter(X_test[gradb_mcidx,0],X_test[gradb_mcidx,1], marker='s', s=60, c=gradb_pred[gradb_mcidx], vmin=0, vmax = 4, cmap='Accent')
#plt.scatter(X_test[gradb_mcidx,0],X_test[gradb_mcidx,1], marker='s', s=10, c=Y_test[gradb_mcidx], vmin=0, vmax = 4, cmap='Accent')

# --------------------
# Random Forest Classifier
# --------------------
rdf = RandomForestClassifier(max_depth=5, n_estimators=30)
rdf.fit(X_train,Y_train)

# Predicting
rdf_pred = rdf.predict(X_test)

# Finding mispredicted samples
rdf_verror = np.asarray([int(rdf_pred[i] != Y_test[i]) for i in range(0,ts)])
rdf_error = np.sum(rdf_verror)
rdf_ccidx = np.where(rdf_verror == 0)
rdf_mcidx = np.where(rdf_verror == 1)

print("----------Random Forest Classfication----------")
print(rdf_error, "misclassified data out of", ts, "(",rdf_error/ts,"%)")

#plt.figure(5)
#plt.scatter(X_train[:,0],X_train[:,1], c=Y_train, vmin=0, vmax = 4, cmap='Accent')
#plt.scatter(X_test[rdf_mcidx,0],X_test[rdf_mcidx,1], marker='s', s=60, c=rdf_pred[rdf_mcidx], vmin=0, vmax = 4, cmap='Accent')
#plt.scatter(X_test[rdf_mcidx,0],X_test[rdf_mcidx,1], marker='s', s=10, c=Y_test[rdf_mcidx], vmin=0, vmax = 4, cmap='Accent')


plt.show()
