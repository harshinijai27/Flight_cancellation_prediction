import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data_flights = pd.read_csv('modifiedFlights3.csv', header=0, low_memory=False)
data_flights.head()
data_flights = data_flights.dropna()
print(data_flights.shape)
print(list(data_flights.columns))
sns.countplot(x='CANCELLED', data=data_flights, palette='hls')
plt.show()
sns.countplot(y='CANCELLATION_REASON', data=data_flights, palette='hls')
plt.show()
sns.countplot(x='AIRLINE', data=data_flights, palette='hls')
plt.show()
data_flights.drop(data_flights.columns[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]], axis=1, inplace=True)
# print(data)
data_flights.groupby('CANCELLED').mean()
print("group by data")
# print(data)
data2_flights = pd.get_dummies(data_flights, columns=['CANCELLED', 'CANCELLATION_REASON', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY',
                                      'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY'])
print("Printing data2")
print(list(data2_flights.columns))
data2_flights.drop(data2_flights.columns[[0, 1, 2, 3, 4, 5, 6, 9, 10]], axis=1, inplace=True)
print(data2_flights)
# print("split up")
X = data2_flights.iloc[:, 1:]
y = data2_flights.iloc[:, 0]
X_train, X_test, y_train, y_test= train_test_split(X, y, random_state=0)
X_train.shape
print(X_train.shape)
print(y_train.shape)
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
from sklearn.decomposition import PCA

X = data2_flights.iloc[:, 1:]
y = data2_flights.iloc[:, 0]
pca = PCA(n_components=2).fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(pca, y, random_state=0)
#logreg=LogisticRegression()
#logreg.fit(X_train, y_train)
plt.figure(dpi=120)
plt.scatter(pca[y.values == 0, 0], pca[y.values == 0, 1], alpha=0.5, label='Cancelled', s=2, color='navy')
plt.scatter(pca[y.values == 1, 0], pca[y.values == 1, 1], alpha=0.5, label='Not Cancelled', s=2, color='darkorange')
plt.legend()
plt.title('Flight Cancellation Prediction')
plt.xlabel('R1')
plt.ylabel('R2')
plt.gca().set_aspect('equal')
plt.show()
#from sklearn.metrics import roc_auc_score
#from sklearn.metrics import roc_curve
#logit_roc_auc = roc_auc_score(y_test, classifier.predict(X_test))
#fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
#plt.figure()
#plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
#plt.plot([0, 1], [0, 1],'r--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic')
#plt.legend(loc="lower right")
#plt.savefig('Log_ROC')
#plt.show()
