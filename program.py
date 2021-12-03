import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings

warnings.filterwarnings('ignore') 
data=pd.read_csv('C:/Users/coco/OneDrive/Desktop/TSL/winequali.csv')

print(data.head())

print(data.describe())

print(data.dtypes)

print(data.isnull().sum()
      
#datacleaning
data.dropna(inplace=True)
typ=pd.get_dummies(data['type'],drop_first=True)
data.drop(['type'],axis=1,inplace=True)
data=pd.concat([data,typ],axis=1)
print(data.isnull().sum())

#datavizuiliz
x=data['alcohol']
y=data['quality']

plt.xlabel('Alcohol')
plt.ylabel('Quality')

plt.pie(x, labels=y, radius=1.1,autopct='%0.01f%%')
      
plt.scatter(x, y)

plt.bar(x, y)
      
plt.show()
      
#traintest
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
print(y_pred)

original=pd.DataFrame(x_test,y_test)
print(original)

#meansquareroot
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,y_pred))

#reggre
print(regressor.score(x_test,y_test))
      
#logisticregg
x=data[['chlorides', 'total sulfur dioxide','density','pH','sulphates','alcohol']]
y=data.quality

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)
print(y_pred)

#confusionmatrix
from sklearn import metrics
cnf_matrix=metrics.confusion_matrix(y_test,y_pred)
print(cnf_matrix)

#heap
sns.heatmap(pd.DataFrame(cnf_matrix),annot=True,fmt='g')
plt.title("confusion matrix")
plt.xlabel("actual label")
plt.ylabel("predicted label")
plt.show()

#accuracy
print(metrics.accuracy_score(y_test,y_pred))      
