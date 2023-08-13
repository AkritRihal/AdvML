import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from perceptron import Perception


data = pd.read_csv("Student_data.csv")

x=data.iloc[:,0:-1].values
y=data.iloc[:, -1].values
# print(x)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train.shape)
print(y_train.shape)


clf=Perception()
clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)
print(accuracy_score(y_test,y_pred))